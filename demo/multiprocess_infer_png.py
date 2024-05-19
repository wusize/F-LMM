import os

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from src.datasets.png import PNGDataset
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN





if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    for subset in ['gt', 'sam', 'conv', 'attn']:
        os.makedirs(os.path.join(args.save_dir, subset), exist_ok=True)

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor = cfg.image_processor
    prompt = cfg.get('prompt', None)

    print(f'Device: {accelerator.device}', flush=True)
    model = BUILDER.build(cfg.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model.eval()

    dataset_params = dict(json_file='data/png_coco_val2017.json',
                          panoptic_json_file='data/coco/annotations/panoptic_val2017.json',
                          panoptic_png_path='data/coco/panoptic_val2017',
                          tokenizer=tokenizer,
                          image_processor=image_processor,
                          prompt_template=prompt_template,
                          local_path='data/coco/val2017',
                          ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/val2017',
                          image2tensor=cfg.get('image2tensor', True),
                          add_image_token=cfg.get('add_image_token', False),
                          image_token=cfg.get('image_token', DEFAULT_IMAGE_TOKEN)
    )
    if prompt is not None:
        dataset_params.update(prompt=prompt)
    png_dataset = PNGDataset(**dataset_params)

    mask_ious = []
    isthing = []
    plural = []
    pixel_accs = []

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()

    data_ids = list(range(len(png_dataset)))[:100]
    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(data_ids) as sub_ids:
        for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
            data_sample = png_dataset[idx]
            with torch.no_grad():
                output = model._forward(data_sample)
            gt_masks = data_sample['gt_masks'].cpu() > 0
            pred_masks = F.interpolate(output['pred_masks'][None].float().cpu(),
                                       size=gt_masks.shape[-2:], mode='bilinear')[0] > 0
            sam_pred_masks = F.interpolate(output['sam_pred_masks'][None].float().cpu(),
                                           size=gt_masks.shape[-2:], mode='bilinear')[0] > 0
            file_name = os.path.basename(data_sample['file_name'])
            

