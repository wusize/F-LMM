import argparse
import torch
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
from demo.utils import colors

import spacy
nlp = spacy.load("en_core_web_sm")
import random
random.shuffle(colors)


def process_noun_chunks(noun_chunks):
    new_noun_chunks = []
    for i in range(len(noun_chunks)):
        noun_chunk = noun_chunks[i]
        if 'image' in noun_chunk.lower():
            continue
        if noun_chunk.lower() in ['it', 'this', 'that', 'those', 'these', 'them',
                                  'he', 'she', 'you', 'i', 'they', 'me', 'her',
                                  'him', 'a', 'what', 'which', 'whose', 'who']:
            continue
        keep = True
        for j in range(len(noun_chunks)):  # de-duplicate
            if i != j and noun_chunk in noun_chunks[j]:
                if len(noun_chunk) < len(noun_chunks[j]) or i > j:
                    keep = False
                    break
        if keep:
            new_noun_chunks.append(noun_chunk)

    return new_noun_chunks


def extract_noun_phrases(output_text):
    doc = nlp(output_text)
    noun_chunks = list(set(chunk.text for chunk in doc.noun_chunks))
    if len(noun_chunks) == 0:
        noun_chunks = [output_text]
    last_end = 0
    noun_chunks = process_noun_chunks(noun_chunks)
    noun_chunks = sorted(noun_chunks, key=lambda x: output_text.find(x))

    noun_chunks = [noun_chunk for noun_chunk in noun_chunks
                   if int(input(f'Ground {noun_chunk}?')) == 1]

    positive_ids = []
    phrases = []
    for noun_chunk in noun_chunks:
        obj_start = output_text.find(noun_chunk)
        if obj_start < last_end:
            continue
        obj_end = obj_start + len(noun_chunk)
        last_end = obj_end
        positive_ids.append((obj_start, obj_end))
        phrases.append(noun_chunk)

    return positive_ids, phrases


def find_interval(intervals, idx):
    for interval_id, (start_id, end_id) in enumerate(intervals):
        if (idx >= start_id) and (idx < end_id):
            return interval_id
    return len(intervals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--image',
                        default='data/coco/val2017/000000000632.jpg', type=str)
    parser.add_argument('--text',
                        default='Where is the shampoo?', type=str)
    parser.add_argument('--checkpoint',
                        default='checkpoints/frozen_deepseek_vl_1_3b_unet_sam_l_iter_95080.pth', type=str)
    parser.add_argument('--use_sam', action='store_true')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor = cfg.image_processor
    prompt = cfg.get('prompt', None)

    model = BUILDER.build(cfg.model)
    state_dict = guess_load_checkpoint(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model._prepare_for_generation(image_processor=image_processor,
                                  prompt_template=prompt_template,
                                  max_thought_tokens=16,
                                  max_new_tokens=512,
                                  lmm_name=cfg.lmm_name,
                                  additional_prompt='')
    model = model.cuda().eval()

    image = Image.open(args.image)
    output = model.single_turn(image, args.text)
    output_ids = output.pop('output_ids').cpu()
    output_text = output.pop('output_text')
    encoded = model.tokenizer(output_text, add_special_tokens=False, return_tensors='pt')
    assert (encoded.input_ids[0] == output_ids).all()
    offsets = encoded.encodings[0].offsets
    str_places, phrases = extract_noun_phrases(output_text)
    positive_ids = []
    for start_id, end_id in str_places:
        start_token_place = find_interval(offsets, start_id)
        end_token_place = max(start_token_place+1, find_interval(offsets, end_id))
        positive_ids.append((start_token_place, end_token_place))
    with torch.no_grad():
        pred_masks, sam_pred_masks = model.ground(image=image, positive_ids=positive_ids, **output)
    if args.use_sam:
        masks = sam_pred_masks.cpu().numpy() > 0
    else:
        masks = pred_masks.cpu().numpy() > 0

    image_np = np.array(image).astype(np.float32)
    for color_id, mask in enumerate(masks):
        image_np[mask] = np.array(colors[color_id]).reshape((1, 1, 3))

    image = Image.fromarray(image_np.astype(np.uint8))
    print(output_text, flush=True)
    print(phrases, flush=True)
    image.save('example.jpg')
