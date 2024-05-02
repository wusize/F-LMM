import argparse
import json
from glob import glob
from accelerate import Accelerator
from tqdm import tqdm
import torch
from accelerate.utils import gather_object
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN
from xtuner.model.utils import guess_load_checkpoint

accelerator = Accelerator()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    accelerator = Accelerator()

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

    json_files = glob("visual_cot/benchmark/*.json")
    for json_file in json_files:
        accelerator.print(f"Processing {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)
        # sync GPUs and start the timer
        accelerator.wait_for_everyone()
        data_ids = list(range(len(data)))
        if args.debug:
            data_ids = data_ids[::100]

        with accelerator.split_between_processes(data_ids) as sub_ids:
            for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
                data_sample = data[idx]
                with torch.no_grad():
                    pred_mask_logits = model.predict(data_sample)
                masks = data_sample['gt_masks'].to(pred_mask_logits.device)
                gt_masks = masks.float().cpu()



