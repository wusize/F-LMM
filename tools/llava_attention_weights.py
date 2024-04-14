import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from frozen_llava.models.llava_next.modeling_llava_next import CustomLlavaNextForConditionalGeneration
from xtuner.utils.templates import PROMPT_TEMPLATE
from transformers import AutoTokenizer
from frozen_llava.datasets.llava_next_image_processor import CustomLlavaNextImageProcessor
from frozen_llava.datasets.gcg import GCGDataset, FlickrForGCGDataset, RefCOCOGForGCGDataset
from frozen_llava.datasets.png import PNGDataset
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--llava_name", default='llava-hf/llava-v1.6-mistral-7b-hf', type=str)
    parser.add_argument("--save_dir", default='data/llava_mistral_7b', type=str)
    parser.add_argument("--merge", default='mean', type=str)
    args = parser.parse_args()
    llava_name = args.llava_name
    os.makedirs(args.save_dir, exist_ok=True)

    if 'mistral' in llava_name:
        prompt_template = PROMPT_TEMPLATE.mistral
    else:
        raise NotImplementedError


    model = CustomLlavaNextForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=llava_name,
        torch_dtype=torch.float16,
        device_map='auto')
    patch_size = model.config.vision_config.patch_size
    
    def _apply_merge(x, dim=1):
        if args.merge == 'mean':
            return x.mean(dim=dim)
        elif args.merge == 'max':
            return x.max(dim=dim).values
        else:
            raise NotImplementedError

    ha_dataset = GCGDataset(json_file='data/GranDf_HA_GCG_train.json',
                            local_path='data/GranDf_HA_images/train',
                            ceph_path='BJ17:S3://wusize/GranDf_HA_images/train',
                            prompt_template=prompt_template,
                            tokenizer=dict(
                                type=AutoTokenizer.from_pretrained,
                                pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                            image_processor=dict(
                                type=CustomLlavaNextImageProcessor.from_pretrained,
                                pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))

    psg_dataset = GCGDataset(json_file='data/OpenPsgGCG_train.json',
                             local_path='data/coco',
                             ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco',
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
        ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2014',
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
        ceph_path='BJ17:S3://wusize/flickr/train',
        prompt_template=prompt_template,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
        image_processor=dict(
            type=CustomLlavaNextImageProcessor.from_pretrained,
            pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'))

    png_dataset = PNGDataset(json_file='data/png_coco_train2017.json',
                             panoptic_json_file='data/coco/annotations/panoptic_train2017.json',
                             panoptic_png_path='data/coco/panoptic_train2017',
                             tokenizer=dict(
                                 type=AutoTokenizer.from_pretrained,
                                 pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                             image_processor=dict(
                                 type=CustomLlavaNextImageProcessor.from_pretrained,
                                 pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                             prompt_template=prompt_template,
                             local_path='data/coco/train2017',
                             ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2017',
                             )


    for name, dataset in zip(['GranDf_HA_GCG_train', 'OpenPsgGCG_train',
                              'RefCOCOg_GCG_train', 'flickr_mergedGT_GCG_train', 'png_coco_train2017'],
                             [ha_dataset, psg_dataset, refcocog_dataset, flickr_dataset, png_dataset]):
        print(f"Process {name}", flush=True)
        dataset_save_dir = os.path.join(args.save_dir, name)
        os.makedirs(dataset_save_dir, exist_ok=True)

        for idx in tqdm(range(len(dataset))):
            data_sample = dataset[idx]
            assert data_sample['pixel_values'].shape[0] > 1
            inputs = dict(input_ids=data_sample['input_ids'][None].to(model.device),
                          mask_ids=data_sample['mask_ids'][None].to(model.device),
                          pixel_values=data_sample['pixel_values'][None].to(device=model.device,
                                                                            dtype=model.dtype),
                          image_sizes=data_sample['image_sizes'][None].to(model.device))
            attention_mask = torch.ones(inputs['input_ids'].shape, device=model.device,
                                        dtype=torch.bool)
            with torch.no_grad():
                outputs = model(**inputs, attention_mask=attention_mask, output_attentions=True)
            fine_image_feature_h, fine_image_feature_w = outputs['image_feature_shapes'][0]
            mask_ids = outputs['mask_ids']
            attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                          for attn in outputs.attentions]
            del outputs

            coarse_image_h, coarse_image_w = data_sample['pixel_values'].shape[2:]
            coarse_image_feature_h, coarse_image_feature_w = (
                coarse_image_h // patch_size, coarse_image_w // patch_size)

            attentions_with_coarse = [
                attn[..., :coarse_image_feature_h*coarse_image_feature_w].view(
                    *attn.shape[:-1], coarse_image_feature_h, coarse_image_feature_w
                ) for attn in attentions]
            attentions_with_fine = [
                attn[..., coarse_image_feature_h*coarse_image_feature_w:].view(
                    *attn.shape[:-1], fine_image_feature_h, fine_image_feature_w+1
                )[..., :-1] for attn in attentions]
            del attentions
            masks = data_sample['masks'].to(model.device)

            attentions_with_coarse_list = []
            attentions_with_fine_list = []
            for mask_id in range(len(masks)):
                matched = mask_ids[0] == mask_id
                assert matched.sum() > 0

                mask_attentions_with_coarse = torch.cat(
                    [_apply_merge(attn[:, matched], dim=1) for attn in attentions_with_coarse])
                mask_attentions_with_fine = torch.cat(
                    [_apply_merge(attn[:, matched], dim=1) for attn in attentions_with_fine])
                attentions_with_coarse_list.append(mask_attentions_with_coarse)
                attentions_with_fine_list.append(mask_attentions_with_fine)
            # print('==================debug================', flush=True)
            attentions_with_coarse = torch.stack(attentions_with_coarse_list)
            attentions_with_fine = torch.stack(attentions_with_fine_list)

            attention_maps = torch.cat([
                F.interpolate(attentions_with_coarse.float(),
                              size=(fine_image_feature_h, fine_image_feature_w), mode='bilinear'),
                F.interpolate(attentions_with_fine.float(),
                              size=(fine_image_feature_h, fine_image_feature_w), mode='bilinear')
            ], dim=1).to(model.dtype)
            del attentions_with_coarse, attentions_with_fine
            gt_masks = F.interpolate(masks.to(attention_maps)[None].float(),
                                     size=(fine_image_feature_h, fine_image_feature_w))[0].to(model.dtype)

            attention_maps_to_save = torch.cat([attention_maps, gt_masks[:, None]], dim=1).half()
            attention_maps_to_save = attention_maps_to_save.detach().cpu().numpy()

            np.save(os.path.join(dataset_save_dir, f'{idx}.jpg'), attention_maps_to_save)
