# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import torch
from PIL import Image
try:
    from petrel_client.client import Client
except:
    Client = None
from xtuner.dataset.utils import expand2square
from xtuner.dataset import LLaVADataset


class CustomLLaVADataset(LLaVADataset):

    def __init__(self, ceph_folder=None, **kwargs):
        super().__init__(**kwargs)
        self.use_ceph = (Client is not None) and (ceph_folder is not None)
        self.FILE_CLIENT = None
        self.ceph_folder = ceph_folder

    def read_image(self, image_file):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_folder, image_file)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            img_bytes = self.FILE_CLIENT.get(image_path)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path)

        return image

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self.read_image(image_file).convert('RGB')
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict
