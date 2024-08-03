import torch
import argparse
import json
import os
import cv2
import numpy as np
from glob import glob
from accelerate import Accelerator
from tqdm import tqdm
from accelerate.utils import gather_object
from mmengine.config import Config
from xtuner.registry import BUILDER
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # import pdb; pdb.set_trace()
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection[1], area_union[1], area_target[1]


def get_mask_from_json(json_path, height, width):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


def draw_mask(image, mask):
    image = np.array(image.convert('RGB')).astype(np.float32)
    image[mask] = image[mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32).reshape(1, 1, 3) * 0.5
    image = image.astype(np.uint8)

    return Image.fromarray(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint',
                        default='checkpoints/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.pth', type=str)
    parser.add_argument('--save_folder', default='', type=str)
    args = parser.parse_args()
    accelerator = Accelerator()
    model_name = os.path.basename(args.config)[:-3]
    os.makedirs(args.save_folder, exist_ok=True)
    args.save_folder = os.path.join(args.save_folder, 'reason_seg')
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
                                  tokenizer=tokenizer,
                                  prompt_template=prompt_template,
                                  max_new_tokens=16,)
    model = model.to(device=accelerator.device)
    model.eval()

    json_files = glob("data/ReasonSeg/val/*.json")

    data_ids = list(range(len(json_files)))
    results = []

    with accelerator.split_between_processes(data_ids) as sub_ids:
        for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
            json_file = json_files[idx]
            image_file = json_file.replace('.json', '.jpg')
            image = Image.open(image_file).convert('RGB')

            gt_mask, instruction, is_sentence = get_mask_from_json(json_file, height=image.height,
                                                                   width=image.width)
            # import pdb; pdb.set_trace()

            if not is_sentence:
                instruction = f"What is {instruction} in this image? "

            instruction += 'Briefly answer the question in a single sentence.'
            answer, mask = model.reason_seg(image=image, instruction=instruction,
                                            answer_prefix='It is')
            intersection, union, _ = intersectionAndUnionGPU(
                mask.clone().long(), torch.from_numpy(gt_mask).to(mask).long(), 2, ignore_index=255
            )
            # import pdb; pdb.set_trace()
            results.append(torch.tensor([intersection.item(), union.item()]))

        results = gather_object(results)

    if accelerator.is_main_process:
        results = torch.stack(results)
        intersections = results[:, 0]
        unions = results[:, 1]
        giou = (intersections / (unions + 1e-12)).mean()
        ciou = intersections.mean() / (unions.mean() + 1e-12)
        print(f'giou: {giou}, ciou: {ciou}', flush=True)
