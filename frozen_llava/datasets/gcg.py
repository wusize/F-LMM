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
from transformers import AutoTokenizer, AutoImageProcessor, LlavaNextProcessor
from frozen_llava.datasets.image_processor import CustomLlavaNextImageProcessor


def in_alphabet(x):
    return x in 'abcdefghijklmnopqrstuvwxyz'



class GCGDataset(Dataset):
    def __init__(self, json_file,
                 image_processor=None, tokenizer=None,
                 ceph_path=None, local_path=None, prompt=''):
        super().__init__()
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.ceph_path = ceph_path
        self.local_path = local_path
        self.FILE_CLIENT = None
        self.use_ceph = (Client is not None) and (ceph_path is not None)
        self.image_processor = CustomLlavaNextImageProcessor.from_pretrained(**image_processor)
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer)
        self.prompt = self.tokenizer.encode(prompt, add_special_tokens=True)


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
        data = self.data[index]
        masks = []
        last_end = 0
        caption = data['caption']
        input_ids = copy.deepcopy(self.prompt)
        mask_ids = [-1] * len(input_ids)
        mask_cnt = 0
        for phrase, obj_info in data['groundings'].items():
            obj_start, obj_end = obj_info['token_positives']
            if obj_start <= 0 or obj_end <= 0:
                continue
            if obj_start < last_end:
                continue
            assert caption[obj_start:obj_end].lower() == phrase.lower()
            while obj_end < len(caption) and in_alphabet(caption[obj_end]):
                obj_end += 1
            assert obj_start >= last_end
            if obj_start > last_end:
                if caption[last_end] in [',', '.', '!', "?", ":", ';']:
                    neg_input = self.tokenizer.encode("debug"+caption[last_end:obj_start].strip(),
                                                      add_special_tokens=False)[1:]
                else:
                    neg_input = self.tokenizer.encode(caption[last_end:obj_start].strip(),
                                                      add_special_tokens=False)
                input_ids += neg_input
                mask_ids += [-1] * len(neg_input)

            # load mask
            mask = np.zeros((data['height'], data['width']), dtype=np.uint8)
            for rle_mask in obj_info['rle_masks']:
                mask += mask_utils.decode(rle_mask)
            masks.append(mask.clip(max=1))

            pos_input = self.tokenizer.encode(caption[obj_start:obj_end].strip(),
                                              add_special_tokens=False)
            input_ids += pos_input
            mask_ids += [mask_cnt] * len(pos_input)

            last_end = obj_end
            mask_cnt += 1
        if mask_cnt == 0:
            return self.__getitem__(random.choice(range(self.__len__())))
        ref = self.prompt + self.tokenizer.encode(caption[:last_end], add_special_tokens=False)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # ref_ids = torch.tensor(ref, dtype=torch.long)

        # assert (input_ids == ref_ids).all()

        # diff = input_ids != ref_ids
        # for id0, id1 in zip(input_ids[diff], ref_ids[diff]):
        #     # the same token can be encoded into different numbers
        #     assert self.tokenizer.decode(id0.item()) == self.tokenizer.decode(id1.item())

        # input_ids = ref_ids

        image = self.read_image(data['file_name'])
        image_data = self.image_processor.preprocess(image)

        pixel_values = torch.from_numpy(image_data['pixel_values'][0])
        meta_data = image_data['meta_datas'][0]

        masks = torch.from_numpy(np.stack(masks))
        resized_masks = F.interpolate(masks[None], size=pixel_values.shape[2:])[0]

        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w =  meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                    padding['before_width']:p_w-padding['after_width']] = masks

        return dict(input_ids=input_ids,
                    pixel_values=pixel_values,
                    resized_masks=resized_masks,   # shape is not kept
                    padded_masks=padded_masks,
                    masks=masks,   # shape is kept
                    image_sizes=torch.tensor(image_data['image_sizes'][0]))




if __name__ == '__main__':
    from frozen_llava.prompt_templates import llava_v1_6_mistral
    from tqdm import tqdm
    dataset = GCGDataset(json_file='data/GranDf_HA_GCG_train.json',
                         local_path='data/GranDf_HA_images/train',
                         prompt=llava_v1_6_mistral.format(input='<image>\nWhat is shown in this image?'),
                         tokenizer=dict(pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                         image_processor=dict(pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))


    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)





















