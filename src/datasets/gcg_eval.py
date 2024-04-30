import os
import io
import copy
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
try:
    from petrel_client.client import Client
except:
    Client = None

import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mmengine.logging import print_log


class GCGEvalDataset(Dataset):
    def __init__(self,
                 caption_json_file,
                 mask_json_file,
                 image_processor=None, tokenizer=None,
                 ceph_path=None, local_path=None, prompt_template=None,
                 prompt='<image>\nPlease give me a detailed description of the image.',
                 image2tensor=True,
                 add_image_token=False,
                 image_token=DEFAULT_IMAGE_TOKEN
                 ):
        super().__init__()
        # prompt += ("If there are multiple instances of a certain object category, "
        #            "you can refer to them as object-1, object-2, object-3, etc. "
        #            "For example, if there are two persons, you can refer to them as person-1, person-2.")
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
        self.prompt_template = prompt_template

        self.caption_data = COCO(caption_json_file)
        self.mask_data = COCO(mask_json_file)
        self.image_ids = self.caption_data.getImgIds()

        self.image2tensor = image2tensor
        self.add_image_token = add_image_token
        self.image_token = image_token

        if self.add_image_token:
            print_log(f"Manually add image token: {self.image_token}")
            special_tokens_dict = {'additional_special_tokens': [self.image_token, ]}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added_toks == 1

        self.image_token_idx = self.tokenizer.encode(self.image_token, add_special_tokens=False)[-1]
        print_log(f"Image token: {self.tokenizer.decode(self.image_token_idx)}")

        self.prompt = self.tokenizer.encode(
            prompt_template['INSTRUCTION'].format(input=prompt),
            add_special_tokens=True)

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

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        data_sample = copy.deepcopy(self.caption_data.imgs[image_id])
        data_sample.update(self.caption_data.imgToAnns[image_id][0])

        masks_list = self.mask_data.imgToAnns.get(image_id, [])
        if len(masks_list) == 0:
            gt_masks = None
        else:
            gt_masks = np.stack(
                [mask_utils.decode(mask_info['segmentation'])
                 for mask_info in masks_list], axis=0)

        image = self.read_image(f'{image_id}.jpg')
        image_data = self.image_processor.preprocess(image)
        # pixel_values = torch.from_numpy(image_data['pixel_values'][0])
        pixel_values = image_data['pixel_values'][0]
        if self.image2tensor:
            pixel_values = torch.from_numpy(pixel_values)

        meta_data = image_data['meta_datas'][0]

        input_ids = torch.tensor(self.prompt, dtype=torch.long)

        if self.add_image_token:
            input_ids[input_ids == self.image_token_idx] = IMAGE_TOKEN_INDEX

        data_sample.update(input_ids=input_ids,
                           pixel_values=pixel_values,
                           meta_data=meta_data,
                           image=image,
                           image_sizes=torch.tensor(image_data['image_sizes'][0]),  # only used for llava-next
                           gt_masks=gt_masks)

        return data_sample

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    from mmengine.config import Config
    cfg = Config.fromfile('configs/fuyu/frozen_fuyu_8b_unet_sam_l.py')
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor = cfg.image_processor
    prompt = cfg.prompt

    dataset = GCGEvalDataset(
        caption_json_file='data/GranDf/val_test/test_gcg_coco_caption_gt.json',
        mask_json_file='data/GranDf/val_test/test_gcg_coco_mask_gt.json',
        image_processor=image_processor,
        tokenizer=tokenizer,
        ceph_path='BJ17:S3://wusize/GranDf_HA_images/val_test',
        local_path='data/GranDf_HA_images/val_test',
        prompt_template=prompt_template,
        prompt=prompt
    )

    for data in dataset:
        print(data.keys())
