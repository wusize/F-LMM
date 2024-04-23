from xtuner.model.utils import guess_load_checkpoint
from src.datasets.gcg_eval import GCGEvalDataset
from accelerate import Accelerator
from accelerate.utils import gather_object
import copy
from time import time
from src.models.gcg_wrapper import GCGWrapper
import torch
import argparse
from tqdm import tqdm
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--prompt',
                        default='<image>\nBriefly describe the objects that appear in this image.', type=str)
    parser.add_argument('--max_new_tokens', default=100, type=int)
    args = parser.parse_args()

    ### Initialize accelerator
    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)

    gcg_wrapper = GCGWrapper(config_file=args.config, max_new_tokens=args.max_new_tokens)

    if accelerator.is_main_process:
        os.makedirs(args.save_path, exist_ok=True)

    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        gcg_wrapper.load_pretrained(state_dict)

    print(f"Start moving model to device: {accelerator.device}", flush=True)
    tik = time()
    gcg_wrapper = gcg_wrapper.to(device=accelerator.device)
    print(f"Finished moving model to device: {accelerator.device}, time used: {time() - tik}", flush=True)
    gcg_wrapper.eval()
    for sub_set in ['val', 'test']:
        if accelerator.is_main_process:
            os.makedirs(f'{args.save_path}/{sub_set}', exist_ok=True)
        dataset = GCGEvalDataset(
                     caption_json_file=f'data/GranDf/val_test/{sub_set}_gcg_coco_caption_gt.json',
                     mask_json_file=f'data/GranDf/val_test/{sub_set}_gcg_coco_mask_gt.json',
                     image_processor=gcg_wrapper.config.image_processor,
                     tokenizer=gcg_wrapper.config.tokenizer,
                     ceph_path='BJ17:S3://wusize/GranDf_HA_images/val_test',
                     local_path='data/GranDf_HA_images/val_test',
                     prompt_template=gcg_wrapper.config.prompt_template,
                     prompt=args.prompt)

        data_ids = list(range(len(dataset)))
        # sync GPUs and start the timer
        accelerator.wait_for_everyone()

        # divide the prompt list onto the available GPUs
        with accelerator.split_between_processes(data_ids) as sub_ids:
            for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
                data_sample = copy.deepcopy(dataset[idx])
                with torch.no_grad():
                    results = gcg_wrapper(data_sample)
                    # Save the inference results
                with open(f"{args.save_path}/{sub_set}/{data_sample['image_id']}.json", 'w') as f:
                    json.dump(results, f)
