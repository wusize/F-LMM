import argparse
import json
import os
from glob import glob
from accelerate import Accelerator
from tqdm import tqdm
from accelerate.utils import gather_object
from mmengine.config import Config
from xtuner.registry import BUILDER
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--image_folder', default='data', type=str)
    parser.add_argument('--version', default='v1', type=str)
    parser.add_argument('--save_folder', default='visual_cot', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    accelerator = Accelerator()
    model_name = os.path.basename(args.config)[:-3]
    os.makedirs(args.save_folder, exist_ok=True)
    args.save_folder = os.path.join(args.save_folder,
                                    f'{model_name}_visual_cot_{args.version}')
    os.makedirs(args.save_folder, exist_ok=True)

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
    state_dict = guess_load_checkpoint(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    accelerator.print(f"Unexpected parameters: {unexpected}")
    model._prepare_for_generation(image_processor=image_processor,
                                  prompt_template=prompt_template,
                                  max_thought_tokens=16,
                                  max_new_tokens=32,
                                  lmm_name=cfg.lmm_name,
                                  additional_prompt='\nAnswer the question using a single word or phrase.')
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

        results = []
        with accelerator.split_between_processes(data_ids) as sub_ids:
            for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
                data_sample = data[idx]
                image = Image.open(os.path.join(args.image_folder, data_sample['image'][0]))
                question = data_sample['conversations'][0]['value'].replace(
                    'Please provide the bounding box coordinate '
                    'of the region that can help you answer the question better.',
                    ''
                )
                question = question.replace('<image>', '').strip()

                thought, box, answer = getattr(model, f'visual_cot_{args.version}')(image, question)
                results.append(dict(thought=thought,
                                    box=box,
                                    answer=answer,
                                    question_id=data_sample['question_id'],
                                    question=question,
                                    image=data_sample['image'][0],
                                    gt=data_sample['conversations'][-1]['value']))
            results = gather_object(results)
        if accelerator.is_main_process:
            accelerator.print(f"Collected {len(results)} result samples from all gpus")

            with open(os.path.join(args.save_folder, os.path.basename(json_file)), 'w') as f:
                json.dump(results, f, indent=4)
