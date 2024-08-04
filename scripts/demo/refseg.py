import random

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
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def draw_mask(image, mask):
    image = np.array(image.convert('RGB')).astype(np.float32)
    image[mask] = image[mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32).reshape(1, 1, 3) * 0.5
    image = image.astype(np.uint8)

    return Image.fromarray(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--image', default='data/ReasonSeg/val', type=str)
    parser.add_argument('--checkpoint',
                        default='checkpoints/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.pth', type=str)
    parser.add_argument('--text', default='The cat next to the dog.', type=str)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template

    model = BUILDER.build(cfg.model)
    image_processor = BUILDER.build(cfg.image_processor)
    tokenizer = BUILDER.build(cfg.tokenizer)
    state_dict = guess_load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)

    image_placeholder = cfg.get('image_placeholder', "<image>\n")
    model = model.cuda()
    model.eval()

    image2tensor = cfg.get('image2tensor', True)
    add_image_token = cfg.get('add_image_token', False)
    image_token = cfg.get('image_token', DEFAULT_IMAGE_TOKEN)

    if add_image_token:
        print(f"Manually add image token: {image_token}")
        special_tokens_dict = {'additional_special_tokens': [image_token, ]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 1

    image_token_idx = tokenizer.encode(image_token, add_special_tokens=False)[-1]
    print(f"Image token: {tokenizer.decode(image_token_idx)}")

    prompt_tokens = tokenizer.encode(
        prompt_template['INSTRUCTION'].format(input=image_placeholder),
        add_special_tokens=True, return_tensors='pt')[0]
    object_tokens = tokenizer.encode(
        args.text, add_special_tokens=False, return_tensors='pt')[0]

    input_ids = torch.cat([prompt_tokens, object_tokens]).cuda()
    image = Image.open(args.image)

    image_data = image_processor.preprocess(image)

    pixel_values = image_data['pixel_values'][0]
    if image2tensor:
        pixel_values = torch.from_numpy(pixel_values)
    meta_data = image_data['meta_datas'][0]

    if add_image_token:
        input_ids[input_ids == image_token_idx] = IMAGE_TOKEN_INDEX

    outputs = model.forward_lmm(dict(pixel_values=pixel_values,
                                     input_ids=input_ids))

    hidden_states = outputs['hidden_states']
    attentions = outputs['attentions']
    import pdb; pdb.set_trace()

