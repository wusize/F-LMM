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

from xtuner.registry import BUILDER
from mmdet.datasets.api_wrappers.coco_api import COCOPanoptic
import mmcv
import io
from mmengine.fileio import get
from panopticapi import utils


class PNGDataset(Dataset):
    def __init__(self,
                 json_file,
                 panoptic_json_file,
                 panoptic_png_path,
                 image_processor=None, tokenizer=None,
                 ceph_path=None, local_path=None, prompt_template=None):
        super().__init__()
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.coco = COCOPanoptic(panoptic_json_file)
        self.panoptic_png_path = panoptic_png_path
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
            prompt_template['INSTRUCTION'].format(input='<image>\nWhat is shown in this image?'),
            add_special_tokens=True)
        self.prompt_template = prompt_template

    @staticmethod
    def _load_segm(segm_path):
        img_bytes = get(segm_path)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        segm_map = utils.rgb2id(pan_png)

        return segm_map

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
        data_sample = self.data[index]
        mask_cnt = 0
        caption_input_ids = []
        mask_ids = [-1]*len(self.prompt)
        mask_segment_ids = []
        mask_infos = []   # record isthing, plural
        image_id = int(data_sample['image_id'])
        annotations = {ann['id']: ann for ann in self.coco.imgToAnns[image_id]}
        for segment in data_sample['segments']:
            segment_input_ids = self.tokenizer.encode(segment['utterance'], add_special_tokens=False)
            caption_input_ids += segment_input_ids
            if len(segment['segment_ids']) == 0:
                mask_ids += [-1] * len(segment_input_ids)
            else:
                mask_ids += [mask_cnt] * len(segment_input_ids)
                mask_segment_ids.append(segment['segment_ids'])
                if not segment['plural']:
                    assert len(segment['segment_ids']) == 1
                    segment_id = int(segment['segment_ids'][0])
                    isthing = self.coco.cats[annotations[segment_id]['category_id']]['isthing']

                else:
                    isthing = 1
                mask_infos.append(dict(plural=segment['plural'],
                                       isthing=isthing > 0))
                # todo: load masks
                mask_cnt += 1

        if mask_cnt == 0:
            return self.__getitem__(random.choice(range(self.__len__())))

        image_info = self.coco.imgs[image_id]
        segm_file = image_info['segm_file']
        segm_map = self._load_segm(os.path.join(self.panoptic_png_path, segm_file))

        masks = []

        for mask_segment_ids_ in mask_segment_ids:
            mask = 0
            for segment_id in mask_segment_ids_:
                mask += (segm_map == int(segment_id)).astype(np.uint8)
            masks.append(np.clip(mask, a_max=1, a_min=0))
        assert len(masks) == mask_cnt

        input_ids = self.prompt + caption_input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        mask_ids = torch.tensor(mask_ids)

        image = self.read_image(image_info['file_name'])
        image_data = self.image_processor.preprocess(image)

        pixel_values = torch.from_numpy(image_data['pixel_values'][0])
        meta_data = image_data['meta_datas'][0]

        masks = torch.from_numpy(np.stack(masks))
        resized_masks = F.interpolate(masks[None], size=pixel_values.shape[2:])[0]

        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                        padding['before_width']:p_w-padding['after_width']] = masks

        return dict(input_ids=input_ids,
                    mask_ids=mask_ids,
                    pixel_values=pixel_values,
                    resized_masks=resized_masks,   # shape is not kept
                    padded_masks=padded_masks,
                    masks=masks,   # shape is kept
                    image_sizes=torch.tensor(image_data['image_sizes'][0]),
                    mask_infos=mask_infos)


if __name__ == '__main__':
    from xtuner.utils.templates import PROMPT_TEMPLATE
    prompt_template = PROMPT_TEMPLATE.mistral
    from transformers import AutoTokenizer
    from transformers import AutoTokenizer
    from frozen_llava.datasets.image_processor import CustomLlavaNextImageProcessor
    from tqdm import tqdm
    dataset = PNGDataset(json_file='data/png_coco_val2017.json',
                         panoptic_json_file='data/coco/annotations/panoptic_val2017.json',
                         panoptic_png_path='data/coco/panoptic_val2017',
                         tokenizer=dict(
                             type=AutoTokenizer.from_pretrained,
                             pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                         image_processor=dict(
                             type=CustomLlavaNextImageProcessor.from_pretrained,
                             pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                         prompt_template=prompt_template,
                         local_path='data/coco/val2017'
                         )

    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)