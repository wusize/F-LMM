import torch
import torch.nn.functional as F
import argparse
from frozen_llava.datasets.png import PNGDataset
from tqdm import tqdm

from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object


accelerator = Accelerator()

def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint_prefix', default='mask_head.', type=str)
    args = parser.parse_args()



    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)



    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor= cfg.image_processor
    llm = cfg.model.model
    mask_head = cfg.model.mask_head
    
    llm = dict(type=llm['type'],
               pretrained_model_name_or_path=llm['pretrained_model_name_or_path'],
               torch_dtype=torch.float16,
               device_map={"": accelerator.process_index},)
    llm = BUILDER.build(llm)
    mask_head = BUILDER.build(mask_head).to(dtype=llm.dtype, device=llm.device)
    llm.eval()
    mask_head.eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        if args.checkpoint_prefix is None:
            mask_head.load_state_dict(state_dict)
        else:
            mask_head.load_state_dict({k.replace(args.checkpoint_prefix, ''): v for k, v in state_dict.items()})

    png_dataset = PNGDataset(json_file='data/png_coco_train2017.json',
                             panoptic_json_file='data/coco/annotations/panoptic_train2017.json',
                             panoptic_png_path='data/coco/panoptic_train2017',
                             tokenizer=tokenizer,
                             image_processor=image_processor,
                             prompt_template=prompt_template,
                             local_path='data/coco/train2017',
                             ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2017',
                             )

    patch_size = llm.config.vision_config.patch_size
    mask_ious = []
    isthing = []
    plural = []
    pixel_accs = []

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()

    data_ids = list(range(len(png_dataset)))[:100]

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(data_ids) as sub_ids:

        for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
            data_sample = png_dataset[idx]
            assert data_sample['pixel_values'].shape[0] > 1
            inputs = dict(input_ids=data_sample['input_ids'][None].to(llm.device),
                          mask_ids=data_sample['mask_ids'][None].to(llm.device),
                          pixel_values=data_sample['pixel_values'][None].to(device=llm.device,
                                                                            dtype=llm.dtype),
                          image_sizes=data_sample['image_sizes'][None].to(llm.device))
            attention_mask = torch.ones(inputs['input_ids'].shape, device=llm.device,
                                        dtype=torch.bool)
            with torch.no_grad():
                outputs = llm(**inputs, attention_mask=attention_mask, output_attentions=True)
            fine_image_feature_h, fine_image_feature_w = outputs['image_feature_shapes'][0]
            mask_ids = outputs['mask_ids']
            attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                          for attn in outputs.attentions]
            del outputs

            coarse_image_h, coarse_image_w = data_sample['pixel_values'].shape[2:]
            coarse_image_feature_h, coarse_image_feature_w = (
                coarse_image_h // patch_size, coarse_image_w // patch_size)

            attentions_with_coarse = [
                attn[..., :coarse_image_feature_h * coarse_image_feature_w].view(
                    *attn.shape[:-1], coarse_image_feature_h, coarse_image_feature_w
                ) for attn in attentions]
            attentions_with_fine = [
                attn[..., coarse_image_feature_h * coarse_image_feature_w:].view(
                    *attn.shape[:-1], fine_image_feature_h, fine_image_feature_w + 1
                )[..., :-1] for attn in attentions]
            del attentions
            masks = data_sample['masks'].to(llm.device)

            attentions_with_coarse_list = []
            attentions_with_fine_list = []
            for mask_id in range(len(masks)):
                matched = mask_ids[0] == mask_id
                assert matched.sum() > 0

                mask_attentions_with_coarse = torch.cat(
                    [torch.mean(attn[:, matched], dim=1) for attn in attentions_with_coarse])
                mask_attentions_with_fine = torch.cat(
                    [torch.mean(attn[:, matched], dim=1) for attn in attentions_with_fine])
                attentions_with_coarse_list.append(mask_attentions_with_coarse)
                attentions_with_fine_list.append(mask_attentions_with_fine)

            attentions_with_coarse = torch.stack(attentions_with_coarse_list)
            attentions_with_fine = torch.stack(attentions_with_fine_list)

            attention_maps = torch.cat([
                F.interpolate(attentions_with_coarse.float(),
                              size=(fine_image_feature_h, fine_image_feature_w), mode='bilinear'),
                F.interpolate(attentions_with_fine.float(),
                              size=(fine_image_feature_h, fine_image_feature_w), mode='bilinear')
            ], dim=1).to(llm.dtype)
            del attentions_with_coarse, attentions_with_fine


            with torch.no_grad():
                pred_masks = mask_head(attention_maps)[:, 0]

            pred_masks = (F.interpolate(pred_masks.to(masks)[None].sigmoid().float(),
                                        size=masks.shape[-2:]) > 0.5)[0].float().cpu()
            gt_masks = masks.float().cpu()

            pixel_acc = [torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks),
                                  gt_masks).to(gt_masks).flatten(1, 2).mean(-1)]

            mask_infos = data_sample['mask_infos']
            sub_mask_ious = [compute_mask_IoU(pred_masks.flatten(1, 2), gt_masks.flatten(1, 2))[-1]]
            sub_isthing = [torch.tensor([mask_info['isthing'] for mask_info in mask_infos])]
            sub_plural = [torch.tensor([mask_info['plural'] for mask_info in mask_infos])]

            sub_mask_ious = gather_object(sub_mask_ious)
            sub_isthing = gather_object(sub_isthing)
            sub_plural = gather_object(sub_plural)
            pixel_acc = gather_object(pixel_acc)

            mask_ious += sub_mask_ious
            isthing += sub_isthing
            plural += sub_plural
            pixel_accs += pixel_acc

    if accelerator.is_main_process:
        mask_ious = torch.cat(mask_ious)
        isthing = torch.cat(isthing)
        plural = torch.cat(plural)

        AA = mask_ious.mean()
        AA_singulars = mask_ious[torch.logical_not(plural)].mean()
        AA_plurals = mask_ious[plural].mean()
        AA_things = mask_ious[isthing].mean()
        AA_stuff = mask_ious[torch.logical_not(isthing)].mean()

        accuracy = (mask_ious > 0.5).float().mean()

        pixel_accs = torch.cat(pixel_accs).mean()


        print(f"aIoU: {AA}, aIoU_singulars: {AA_singulars}, aIoU_plurals: {AA_plurals}, "
              f"aIoU_things: {AA_things}, aIoU_stuff: {AA_stuff}, aAcc@0.5: {accuracy}, "
              f"pixel_accs: {pixel_accs}", flush=True)
