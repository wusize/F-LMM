import os
import io
import json
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import random
try:
    from petrel_client.client import Client
except:
    Client = None

import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from xtuner.registry import BUILDER
from typing import Dict, Sequence
from tqdm import tqdm
from torch.utils.data import ConcatDataset


def concat_datasets(datasets_list):
    datasets_list = [BUILDER.build(dataset_) for dataset_ in datasets_list]
    return ConcatDataset(datasets_list)


def gcg_collate_fn(instances: Sequence[Dict]):
    # return instances
    # all list
    return {'data': instances, 'data_samples': None}
    # keys = instances[0].keys()
    # return {k: [inst[k] for inst in instances] for k in keys}


class GCGDataset(Dataset):
    def __init__(self, json_file,
                 image_processor=None, tokenizer=None,
                 ceph_path=None, local_path=None, prompt_template=None,
                 prompt='<image>\nWhat is shown in this image?'):
        super().__init__()
        self._load_annotations(json_file)
        self.ceph_path = ceph_path
        self.local_path = local_path
        self.FILE_CLIENT = None
        self.use_ceph = (Client is not None) and (ceph_path is not None)
        if isinstance(tokenizer, dict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        if isinstance(image_processor, dict):
           self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.prompt = self.tokenizer.encode(
            prompt_template['INSTRUCTION'].format(input=prompt),
            add_special_tokens=True)
        self.prompt_template = prompt_template

        special_tokens_dict = {'additional_special_tokens': ['<mask>', '</mask>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 2

    def _load_annotations(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    @property
    def mask_start_id(self):
        return self.tokenizer.added_tokens_encoder['<mask>']

    @property
    def mask_end_id(self):
        return self.tokenizer.added_tokens_encoder['</mask>']

    def __len__(self):
        return len(self.data)

    def read_image(self, image_file):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_path, image_file)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            img_bytes = self.FILE_CLIENT.get(image_path)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            image_path = os.path.join(self.local_path, image_file)
            image = Image.open(image_path)

        return image

    def __getitem__(self, index):
        sample_data = self.data[index]
        masks = []
        last_end = 0
        mask_cnt = 0

        caption = copy.deepcopy(sample_data['caption'])
        new_caption = ''
        for phrase, obj_info in sample_data['groundings'].items():
            obj_start, obj_end = obj_info['token_positives']
            if obj_start < 0 or obj_end <= 0:
                continue
            if obj_start < last_end:
                continue
            assert caption[obj_start:obj_end].lower() == phrase.lower()

            new_caption += f"{caption[last_end:obj_start].strip()}<mask>{caption[obj_start:obj_end]}</mask>"
            # load mask
            mask = np.zeros((sample_data['height'], sample_data['width']), dtype=np.uint8)
            for rle_mask in obj_info['rle_masks']:
                mask += mask_utils.decode(rle_mask)
            masks.append(mask.clip(max=1))
            last_end = obj_end
            mask_cnt += 1
        if mask_cnt == 0:
            return self.__getitem__(random.choice(range(self.__len__())))

        input_ids = self.prompt + self.tokenizer.encode(new_caption, add_special_tokens=False)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Fixme: when special tokens are followed by punctuations, the behaviour is different
        # input_ids = input_ids[input_ids != self.tokenizer.encode(',', add_special_tokens=False)[-1]]

        final_input_ids = []
        mask_ids = []
        mask_start_ids = torch.where(input_ids == self.mask_start_id)[0]
        mask_end_ids = torch.where(input_ids == self.mask_end_id)[0]
        assert len(mask_end_ids) == len(mask_start_ids)
        assert len(mask_end_ids) == mask_cnt

        last_id = 0
        for mask_id, (mask_start_id, mask_end_id) in enumerate(zip(mask_start_ids, mask_end_ids)):
            if last_id < mask_start_id:
                final_input_ids.append(input_ids[last_id:mask_start_id])
                mask_ids += [-1] * (mask_start_id - last_id)

            final_input_ids.append(input_ids[mask_start_id+1:mask_end_id])
            mask_ids += [mask_id] * (mask_end_id-1-mask_start_id)
            last_id = mask_end_id + 1

        final_input_ids = torch.cat(final_input_ids)
        mask_ids = torch.tensor(mask_ids)

        image = self.read_image(sample_data['file_name'])
        image_data = self.image_processor.preprocess(image)

        pixel_values = torch.from_numpy(image_data['pixel_values'][0])
        meta_data = image_data['meta_datas'][0]

        masks = torch.from_numpy(np.stack(masks))
        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                        padding['before_width']:p_w-padding['after_width']] = masks

        return dict(input_ids=final_input_ids,
                    mask_ids=mask_ids,
                    pixel_values=pixel_values,
                    padded_masks=padded_masks,
                    masks=masks,   # shape is kept
                    image_sizes=torch.tensor(image_data['image_sizes'][0]),   # only used for llava-next
                    image=image,
                    meta_data=meta_data)


class RefCOCOGForGCGDataset(GCGDataset):
    def __getitem__(self, index):
        data_sample = list(self.data[index].values())[0]
        segmentations = []
        last_end = 0
        mask_cnt = 0
        caption = copy.deepcopy(data_sample['caption'])
        new_caption = ''
        for ref in data_sample['refs']:
            if ref['sentence'] in caption.lower():
                obj_start = caption.lower().find(ref['sentence'])
                if obj_start == -1:
                    continue
                obj_end = obj_start + len(ref['sentence'])
                segmentations.append(ref['segmentation'])

                new_caption += f"{caption[last_end:obj_start].strip()}<mask>{caption[obj_start:obj_end]}</mask>"
                last_end = obj_end
                mask_cnt += 1

        if mask_cnt == 0:
            return self.__getitem__(random.choice(range(self.__len__())))

        input_ids = self.prompt + self.tokenizer.encode(new_caption, add_special_tokens=False)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Fixme: when special tokens are followed by punctuations, the behaviour is different
        # input_ids = input_ids[input_ids != self.tokenizer.encode(',', add_special_tokens=False)[-1]]

        final_input_ids = []
        mask_ids = []
        mask_start_ids = torch.where(input_ids == self.mask_start_id)[0]
        mask_end_ids = torch.where(input_ids == self.mask_end_id)[0]
        assert len(mask_end_ids) == len(mask_start_ids)
        assert len(mask_end_ids) == mask_cnt

        last_id = 0
        for mask_id, (mask_start_id, mask_end_id) in enumerate(zip(mask_start_ids, mask_end_ids)):
            if last_id < mask_start_id:
                final_input_ids.append(input_ids[last_id:mask_start_id])
                mask_ids += [-1] * (mask_start_id - last_id)

            final_input_ids.append(input_ids[mask_start_id+1:mask_end_id])
            mask_ids += [mask_id] * (mask_end_id-1-mask_start_id)
            last_id = mask_end_id + 1

        final_input_ids = torch.cat(final_input_ids)
        mask_ids = torch.tensor(mask_ids)

        image = self.read_image(data_sample['img_file_name'])
        height, width = image.height, image.width

        masks = []

        for segmentation in segmentations:
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in segmentation:
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask.clip(max=1))

        image_data = self.image_processor.preprocess(image)
        pixel_values = torch.from_numpy(image_data['pixel_values'][0])
        meta_data = image_data['meta_datas'][0]

        masks = torch.from_numpy(np.stack(masks))
        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                        padding['before_width']:p_w-padding['after_width']] = masks

        return dict(input_ids=final_input_ids,
                    mask_ids=mask_ids,
                    pixel_values=pixel_values,
                    padded_masks=padded_masks,
                    masks=masks,   # shape is kept
                    image_sizes=torch.tensor(image_data['image_sizes'][0]),
                    image=image,
                    meta_data=meta_data)


class FlickrForGCGDataset(GCGDataset):
    def _load_annotations(self, ann_file):
        # Load annotations and filter out images with very short captions
        coco = COCO(ann_file)
        self.data = []
        for image_id, anns in tqdm(coco.imgToAnns.items()):
            image_info = coco.imgs[image_id]
            caption = image_info['caption']
            if len(caption.split(' ')) < 3:
                continue
            height = int(image_info['height'])
            width = int(image_info['width'])
            if height <= 32 or width <= 32:
                continue

            file_name = image_info['file_name'].split('_')[-1]

            self.data.append(dict(caption=caption,
                                  file_name=file_name,
                                  width=width,
                                  height=height,
                                  annotations=anns)
                             )

    def __getitem__(self, index):
        sample_data = self.data[index]
        masks = []
        mask_cnt = 0
        new_caption = ''
        tokens_positive_list = []
        tokens_positive_cnt = 0
        tokens_positive2mask = {}
        for annotation in sample_data['annotations']:
            for tokens_positive in annotation['tokens_positive']:
                tokens_positive_list.append(tokens_positive)
                tokens_positive2mask[tokens_positive_cnt] = mask_cnt
                tokens_positive_cnt += 1
            # load mask
            masks.append(mask_utils.decode(annotation['sam_mask']))
            mask_cnt += 1
        assert tokens_positive_cnt >= mask_cnt
        if mask_cnt == 0:
            return self.__getitem__(random.choice(range(self.__len__())))

        tokens_positive_order = sorted(range(tokens_positive_cnt),
                                       key=lambda x: tokens_positive_list[x][0])
        last_end = 0
        caption = copy.deepcopy(sample_data['caption'])

        for tokens_positive_idx in tokens_positive_order:
            obj_start, obj_end = tokens_positive_list[tokens_positive_idx]
            assert obj_start >= last_end
            new_caption += f"{caption[last_end:obj_start].strip()}<mask>{caption[obj_start:obj_end]}</mask>"
            last_end = obj_end

        input_ids = self.prompt + self.tokenizer.encode(new_caption, add_special_tokens=False)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Fixme: when special tokens are followed by punctuations, the behaviour is different
        # input_ids = input_ids[input_ids != self.tokenizer.encode(',', add_special_tokens=False)[-1]]

        final_input_ids = []
        mask_ids = []
        obj_start_ids = torch.where(input_ids == self.mask_start_id)[0]
        mask_end_ids = torch.where(input_ids == self.mask_end_id)[0]
        assert len(mask_end_ids) == len(obj_start_ids)
        assert len(mask_end_ids) == tokens_positive_cnt

        last_id = 0
        for tokens_positive_order_idx, (obj_start_id, obj_end_id) in enumerate(zip(obj_start_ids, mask_end_ids)):
            if last_id < obj_start_id:
                final_input_ids.append(input_ids[last_id:obj_start_id])
                mask_ids += [-1] * (obj_start_id - last_id)
            tokens_positive_idx = tokens_positive_order[tokens_positive_order_idx]
            mask_id = tokens_positive2mask[tokens_positive_idx]
            final_input_ids.append(input_ids[obj_start_id+1:obj_end_id])
            mask_ids += [mask_id] * (obj_end_id-1-obj_start_id)
            last_id = obj_end_id + 1

        final_input_ids = torch.cat(final_input_ids)
        mask_ids = torch.tensor(mask_ids)

        image = self.read_image(sample_data['file_name'])
        image_data = self.image_processor.preprocess(image)

        pixel_values = torch.from_numpy(image_data['pixel_values'][0])
        meta_data = image_data['meta_datas'][0]

        masks = torch.from_numpy(np.stack(masks))

        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                        padding['before_width']:p_w-padding['after_width']] = masks

        return dict(input_ids=final_input_ids,
                    mask_ids=mask_ids,
                    pixel_values=pixel_values,
                    padded_masks=padded_masks,
                    masks=masks,   # shape is kept
                    image_sizes=torch.tensor(image_data['image_sizes'][0]),
                    image=image,
                    meta_data=meta_data)


# TODO: use MUSE dataset
class MuseForGCGDataset(GCGDataset):
    def __init__(self, prompt_template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_template = prompt_template


if __name__ == '__main__':
    dataset_list = []
    from xtuner.utils.templates import PROMPT_TEMPLATE
    from transformers import AutoTokenizer
    from src.datasets.llava_next_processors import CustomLlavaNextImageProcessor
    prompt_template = PROMPT_TEMPLATE.mistral
    ha_dataset = GCGDataset(json_file='data/GranDf_HA_GCG_train.json',
                            local_path='data/GranDf_HA_images/train',
                            prompt_template=prompt_template,
                            tokenizer=dict(
                                type=AutoTokenizer.from_pretrained,
                                pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                            image_processor=dict(
                                type=CustomLlavaNextImageProcessor.from_pretrained,
                                pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))
    
    psg_dataset = GCGDataset(json_file='data/OpenPsgGCG_train.json',
                             local_path='data/coco',
                             prompt_template=prompt_template,
                             tokenizer=dict(
                                 type=AutoTokenizer.from_pretrained,
                                 pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                             image_processor=dict(
                                 type=CustomLlavaNextImageProcessor.from_pretrained,
                                 pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))

    refcocog_dataset = RefCOCOGForGCGDataset(
        json_file='data/RefCOCOg_GCG_train.json',
        local_path='data/coco/train2014',
        prompt_template=prompt_template,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
        image_processor=dict(
            type=CustomLlavaNextImageProcessor.from_pretrained,
            pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))

    flickr_dataset = FlickrForGCGDataset(
        json_file='data/flickr_mergedGT_GCG_train.json',
        local_path='data/flickr/train',
        prompt_template=prompt_template,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
        image_processor=dict(
            type=CustomLlavaNextImageProcessor.from_pretrained,
            pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))

    dataset = ConcatDataset([ha_dataset, psg_dataset, refcocog_dataset, flickr_dataset])

    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
