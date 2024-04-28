# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, Union, Tuple, List
from PIL import Image
import mmengine.fileio as fileio
from mmengine.logging import print_log
import io
from mmcv.transforms import LoadImageFromFile, BaseTransform
from xtuner.registry import BUILDER
from xtuner.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import torch
import torch.nn.functional as F
import copy

try:
    from petrel_client.client import Client
except:
    Client = None


class PILLoadImageFromFile(LoadImageFromFile):
    def __init__(self, **kwargs):
        backend_args = kwargs.pop('backend_args', None)
        if Client is None:
            backend_args = None
        super().__init__(backend_args=backend_args, **kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        results['img'] = img
        results['img_shape'] = (img.height, img.width)
        results['ori_shape'] = (img.height, img.width)
        return results


class RefCOCO2PNG(BaseTransform):
    def __init__(self,
                 image_processor=None,
                 tokenizer=None,
                 prompt_template=None,
                 prompt='<image>\nWhat is shown in this image?',
                 concat=True,
                 image2tensor=True,
                 add_image_token=False):
        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.prompt = self.tokenizer.encode(
            prompt_template['INSTRUCTION'].format(input=prompt),
            add_special_tokens=True)
        self.prompt_template = prompt_template
        self.concat = concat
        self.image2tensor = image2tensor

        self.add_image_token = add_image_token
        if add_image_token:
            special_tokens_dict = {'additional_special_tokens': ['<image>', ]}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added_toks == 1

        self.image_token_idx = self.tokenizer.encode('<image>', add_special_tokens=False)[-1]
        print_log(f"Image token: {self.tokenizer.decode(self.image_token_idx)}")

    def transform(self, results):
        if self.concat:
            return self.transform_concat(results)
        else:
            return self.transform_split(results)

    def transform_split(self, results):
        all_results = []
        for inst_id, instant_text in enumerate(results['text']):
            new_results = copy.deepcopy(results)
            new_results['text'] = [instant_text]
            new_results['gt_masks'] = results['gt_masks'][inst_id:inst_id+1]
            all_results.append(self.transform_concat(new_results))

        return all_results

    def transform_concat(self, results: dict):

        caption_input_ids = []
        mask_ids = [-1] * len(self.prompt)
        split_token_id = self.tokenizer.encode('.', add_special_tokens=False)[-1]

        for inst_id, instant_text in enumerate(results['text']):
            segment_input_ids = self.tokenizer.encode(instant_text, add_special_tokens=False)
            caption_input_ids += segment_input_ids
            mask_ids += [inst_id] * len(segment_input_ids)

            caption_input_ids.append(split_token_id)
            mask_ids.append(-1)

        input_ids = self.prompt + caption_input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        mask_ids = torch.tensor(mask_ids)

        image = results['img']
        image_data = self.image_processor.preprocess(image)

        pixel_values = image_data['pixel_values'][0]
        if self.image2tensor:
            pixel_values = torch.from_numpy(pixel_values)
        meta_data = image_data['meta_datas'][0]

        assert len(results['gt_masks'].masks) == len(results['text'])
        mask_cnt = len(results['text'])

        masks = torch.from_numpy(results['gt_masks'].masks).float()

        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        gt_masks = masks.clone()
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h - padding['after_height'],
                        padding['before_width']:p_w - padding['after_width']] = masks

        # todo: add labels
        prompt_len = len(self.prompt)
        labels = torch.ones_like(input_ids) * IGNORE_INDEX
        labels[prompt_len:] = input_ids[prompt_len:]

        if self.add_image_token:
            import pdb; pdb.set_trace()
            input_ids[input_ids == self.image_token_idx] = IMAGE_TOKEN_INDEX

        return dict(input_ids=input_ids,
                    mask_ids=mask_ids,
                    pixel_values=pixel_values,
                    padded_masks=padded_masks,
                    masks=masks,  # shape is kept
                    gt_masks=gt_masks,
                    image_sizes=torch.tensor(image_data['image_sizes'][0]),
                    image=image,
                    meta_data=meta_data,
                    labels=labels)


if __name__ == '__main__':
    from mmdet.datasets import RefCocoDataset
    from mmengine.config import Config
    from mmdet.datasets.transforms import LoadAnnotations

    cfg = Config.fromfile('configs/fuyu/frozen_fuyu_8b_unet_sam_l.py')
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor = cfg.image_processor
    prompt = cfg.get('prompt', None)

    refcoco2png_params = dict(
        type=RefCOCO2PNG,
        image_processor=image_processor,
        tokenizer=tokenizer,
        prompt_template=prompt_template,

    )
    if prompt is not None:
        refcoco2png_params.update(prompt=prompt)

    test_pipeline = [
        dict(type=PILLoadImageFromFile, backend_args=None),
        dict(
            type=LoadAnnotations,
            with_mask=True,
            with_bbox=False,
            with_seg=False,
            with_label=False),
        refcoco2png_params
    ]

    dataset = RefCocoDataset(
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        text_mode='select_first',
        pipeline=test_pipeline,
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='val'
    )


    for data in dataset:
        print(data.keys())
