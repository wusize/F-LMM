import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from flmm.datasets.png import PNGDataset
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN
from PIL import Image
from scripts.demo.utils import colors

accelerator = Accelerator()


def draw_mask(image, mask, c=(255, 0, 0)):
    image = np.array(image.convert('RGB')).astype(np.float32)
    image[mask] = image[mask] * 0.5 + np.array(c, dtype=np.float32).reshape(1, 1, 3) * 0.5
    image = image.astype(np.uint8)

    return Image.fromarray(image)


def average_accuracy(ious):
    ious = ious.cpu().numpy()
    accuracy = []
    average_acc = 0
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (ious >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)

        accuracy.append(a)
    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])):
        average_acc += (np.abs(t[1] - t[0])) * accuracy[i]

    return average_acc


def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)


def mask2box(mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    return np.array([x0, y0, x1, y1])


def mask2point(mask, image_h, image_w):
    h, w = mask.shape
    ys, xs = np.where(mask)
    ys, xs = (image_h * (ys.astype(np.float32) + 0.5) / h,
              image_w * (xs.astype(np.float32) + 0.5) / w)
    return np.stack([xs, ys], axis=1)


def mask2logits(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask > 0] = 1 - eps
    logits[mask < 1] = eps
    logits = inv_sigmoid(logits)

    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

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

    dataset_params = dict(json_file='data/coco/annotations/png_coco_val2017.json',
                          panoptic_json_file='data/coco/annotations/panoptic_val2017.json',
                          panoptic_png_path='data/coco/annotations/panoptic_val2017',
                          tokenizer=tokenizer,
                          image_processor=image_processor,
                          prompt_template=prompt_template,
                          local_path='data/coco/val2017',
                          # ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/val2017',
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

    data_ids = list(range(len(png_dataset)))
    if args.debug:
        data_ids = data_ids[::100]
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(data_ids) as sub_ids:

        for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
            data_sample = png_dataset[idx]
            with torch.no_grad():
                pred_mask_logits = model.predict(data_sample)
            masks = data_sample['gt_masks'].to(pred_mask_logits.device)
            gt_masks = masks.float().cpu()

            pred_masks = F.interpolate(pred_mask_logits[None].float().sigmoid(),
                                       size=masks.shape[-2:], mode='bilinear')[0].cpu()
            pred_masks = (pred_masks > 0.5).float()

            assert pred_masks.shape == gt_masks.shape
            mask_cnt = pred_masks.shape[0]
            # import pdb; pdb.set_trace()
            meta_data = png_dataset.data[idx]
            if args.output is not None:
                with open(os.path.join(args.output, f"{meta_data['image_id']}.json"), 'w') as f:
                    json.dump(fp=f, obj=meta_data)

                image = data_sample['image']
                for mask_cnt in range(pred_masks.shape[0]):
                    image = draw_mask(image=image, mask=pred_masks[mask_cnt].numpy() > 0,
                                      c=colors[mask_cnt % len(colors)])
                image.save(os.path.join(args.output, f"{meta_data['image_id']}.jpg"))

            mask_infos = data_sample['mask_infos']
            sub_mask_ious = [compute_mask_IoU(pred_masks.flatten(1, 2), gt_masks.flatten(1, 2))[-1]]
            sub_isthing = [torch.tensor([mask_info['isthing'] for mask_info in mask_infos])]
            sub_plural = [torch.tensor([mask_info['plural'] for mask_info in mask_infos])]
            pixel_acc = [torch.eq(pred_masks, gt_masks).float().flatten(1, 2).mean(-1)]

            mask_ious += sub_mask_ious
            isthing += sub_isthing
            plural += sub_plural
            pixel_accs += pixel_acc

        mask_ious = gather_object(mask_ious)
        isthing = gather_object(isthing)
        plural = gather_object(plural)
        pixel_accs = gather_object(pixel_accs)

    if accelerator.is_main_process:
        mask_ious = torch.cat(mask_ious)
        isthing = torch.cat(isthing)
        plural = torch.cat(plural)

        AA = average_accuracy(mask_ious)
        AA_singulars = average_accuracy(mask_ious[torch.logical_not(plural)])
        AA_plurals = average_accuracy(mask_ious[plural])
        AA_things = average_accuracy(mask_ious[isthing])
        AA_stuff = average_accuracy(mask_ious[torch.logical_not(isthing)])

        accuracy = (mask_ious > 0.5).float().mean()

        pixel_accs = torch.cat(pixel_accs).mean()

        print(f"aIoU: {AA}, aIoU_singulars: {AA_singulars}, aIoU_plurals: {AA_plurals}, "
              f"aIoU_things: {AA_things}, aIoU_stuff: {AA_stuff}, aAcc@0.5: {accuracy}, "
              f"pixel_accs: {pixel_accs}", flush=True)
