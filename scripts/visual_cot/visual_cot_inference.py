import argparse
import json
import os
import numpy as np
from glob import glob
from accelerate import Accelerator
from tqdm import tqdm
from accelerate.utils import gather_object
from mmengine.config import Config
from xtuner.registry import BUILDER
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
import mmcv
from torch.nn.functional import interpolate

def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def draw_box(image, box):
    image = np.array(image.convert('RGB'))
    image = mmcv.imshow_bboxes(img=image,
                               bboxes=np.array(box).reshape(1, 4),
                               colors=(255, 0, 0),
                               thickness=2,
                               show=False)

    return Image.fromarray(image)


def draw_mask(image, mask):
    image = np.array(image.convert('RGB')).astype(np.float32)
    image[mask] = image[mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32).reshape(1, 1, 3) * 0.5
    image = image.astype(np.uint8)
    image = mmcv.imshow_bboxes(img=image,
                               bboxes=np.array(box).reshape(1, 4),
                               colors=(255, 0, 0),
                               thickness=2,
                               show=False)

    return Image.fromarray(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint',
                        default='checkpoints/frozen_deepseek_vl_1_3b_unet_sam_l_iter_95080.pth', type=str)
    parser.add_argument('--image_folder', default='data', type=str)
    parser.add_argument('--version', default='v1', type=str)
    parser.add_argument('--save_folder', default='visual_cot', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--discard_sam', action='store_true')
    parser.add_argument('--box_scale', default=1.0, type=float)
    args = parser.parse_args()
    accelerator = Accelerator()
    model_name = os.path.basename(args.config)[:-3]
    os.makedirs(args.save_folder, exist_ok=True)
    args.save_folder = os.path.join(args.save_folder,
                                    f'{model_name}_visual_cot_{args.version}')
    if args.debug:
        args.save_folder += 'debug'
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
                                  additional_prompt='\nAnswer the question using a single word or phrase.',
                                  box_scale=args.box_scale,
                                  use_sam=not args.discard_sam)
    model = model.to(device=accelerator.device)
    model.eval()

    json_files = glob("scripts/visual_cot/benchmark/*.json")
    for json_file in json_files:
        accelerator.print(f"Processing {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)
        # sync GPUs and start the timer
        accelerator.wait_for_everyone()
        data_ids = list(range(len(data)))
        if args.debug:
            data_ids = data_ids[::50]

        results = []
        # ious = []
        os.makedirs(os.path.join(args.save_folder, f'{os.path.basename(json_file)[:-4]}'), exist_ok=True)
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
                gt_bbox = data_sample['image'][1].split('###')[-1].replace('[', '').replace(']', '')
                gt_bbox = [int(x) for x in gt_bbox.split(',')]
                thought, box, answer, mask = getattr(model, f'visual_cot_{args.version}')(image, question, gt_bbox)
                # iou = get_iou(box, gt_bbox)
                # ious.append(iou)
                image = draw_box(image, box)
                if mask is not None:
                    mask = interpolate(mask[None, None].float(), size=(image.height, image.width), mode='bilinear')
                    mask = (mask[0, 0] > 0.0).cpu().numpy()
                    image = draw_mask(image, mask)
                image.save(os.path.join(args.save_folder,
                                        f"{os.path.basename(json_file)[:-4]}/{os.path.basename(data_sample['image'][0])}"))
                results.append(dict(thought=thought,
                                    box=box,
                                    gt_bbox=gt_bbox,
                                    # iou=iou,
                                    answer=answer,
                                    question_id=data_sample['question_id'],
                                    question=question,
                                    image=data_sample['image'][0],
                                    gt=data_sample['conversations'][-1]['value']))
            results = gather_object(results)
            # ious = gather_object(ious)
        if accelerator.is_main_process:
            accelerator.print(f"Collected {len(results)} result samples from all gpus")
            # accelerator.print(f"Average IoU on {json_file}: {sum(ious) / len(ious)}")
            with open(os.path.join(args.save_folder, os.path.basename(json_file)), 'w') as f:
                json.dump(results, f, indent=4)
