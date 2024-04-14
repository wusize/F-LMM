import numpy as np
import torch
import torch.nn.functional as F
import argparse
from frozen_llava.datasets.png import PNGDataset
from tqdm import tqdm
from functools import partial
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object


accelerator = Accelerator()


def apply_merge(x, merge_type='mean', dim=1):
    if merge_type == 'mean':
        return x.mean(dim=dim)
    elif merge_type == 'max':
        return x.max(dim=dim).values
    else:
        raise NotImplementedError

def average_accuracy(ious):
    ious = ious.cpu().numpy()
    accuracy = []
    average_acc = 0
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (ious >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)

        accuracy.append(a)
    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])):
        average_acc += (np.abs(t[1] - t[0])) * accuracy[i]

    return average_acc

def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)


def mask2box(mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    return np.array([x0, y0, x1, y1])


def mask2point(mask, image_h, image_w):
    h, w = mask.shape
    ys, xs = np.where(mask)
    ys, xs = (image_h * (ys.astype(np.float32) + 0.5) / h,
              image_w * (xs.astype(np.float32) + 0.5) / w)
    return np.stack([xs, ys], axis=1)


def mask2logits(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask > 0] = 1 - eps
    logits[mask < 1] = eps
    logits = inv_sigmoid(logits)

    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint_prefix', default='mask_head.', type=str)
    parser.add_argument('--sam_model', default=None, type=str)
    parser.add_argument('--sam_checkpoint', default=None, type=str)
    args = parser.parse_args()
    print(f'preserve_logits: {args.preserve_logits}', flush=True)

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
    merge_func = partial(apply_merge, merge_type=cfg.model.get('merge', 'mean'))
    llm = dict(type=llm['type'],
               pretrained_model_name_or_path=llm['pretrained_model_name_or_path'],
               torch_dtype=torch.float16,
               device_map={"": accelerator.process_index},)
    llm = BUILDER.build(llm)
    mask_head = BUILDER.build(mask_head).to(dtype=llm.dtype, device=llm.device)
    if args.sam_model is not None:
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(device=llm.device)
        sam.eval()
        sam_predictor = SamPredictor(sam)
    else:
        sam_predictor = None

    llm.eval()
    mask_head.eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        if args.checkpoint_prefix is None:
            mask_head.load_state_dict(state_dict)
        else:
            mask_head.load_state_dict({k.replace(args.checkpoint_prefix, ''): v for k, v in state_dict.items()})

    png_dataset = PNGDataset(json_file='data/png_coco_val2017.json',
                             panoptic_json_file='data/coco/annotations/panoptic_val2017.json',
                             panoptic_png_path='data/coco/panoptic_val2017',
                             tokenizer=tokenizer,
                             image_processor=image_processor,
                             prompt_template=prompt_template,
                             local_path='data/coco/val2017',
                             ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/val2017',
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
            masks = data_sample['gt_masks'].to(llm.device)

            attentions_with_coarse_list = []
            attentions_with_fine_list = []
            for mask_id in range(len(masks)):
                matched = mask_ids[0] == mask_id
                assert matched.sum() > 0

                mask_attentions_with_coarse = torch.cat(
                    [merge_func(attn[:, matched], dim=1) for attn in attentions_with_coarse])
                mask_attentions_with_fine = torch.cat(
                    [merge_func(attn[:, matched], dim=1) for attn in attentions_with_fine])
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
            gt_masks = masks.float().cpu()

            with torch.no_grad():
                pred_mask_logits = mask_head(attention_maps)[:, 0]

            if sam_predictor is not None:
                image = np.array(data_sample['image'].convert('RGB'))
                sam_predictor.set_image(image)
            pred_masks = F.interpolate(pred_mask_logits[None].float().sigmoid(),
                                       size=masks.shape[-2:], mode='bilinear')[0].cpu()
            pred_masks = (pred_masks > 0.5).float()

            if sam_predictor is not None:
                # image = np.array(data_sample['image'].convert('RGB'))
                # sam_predictor.set_image(image)
                sam_masks = []

                for pred_mask in pred_masks:
                    prompt_box = mask2box(pred_mask.numpy())
                    sam_outputs = sam_predictor.predict(
                        box=prompt_box,
                    )
                    candidate_masks = torch.from_numpy(sam_outputs[0]).float()
                    candidate_ious = compute_mask_IoU(candidate_masks.view(3, -1),
                                                      pred_mask.view(1, -1))[-1]
                    sam_mask = candidate_masks[candidate_ious.argmax()]
                    sam_masks.append(sam_mask)

                pred_masks = torch.stack(sam_masks).float()

            assert pred_masks.shape == gt_masks.shape
            mask_cnt = pred_masks.shape[0]

            mask_infos = data_sample['mask_infos']
            sub_mask_ious = [compute_mask_IoU(pred_masks.flatten(1, 2), gt_masks.flatten(1, 2))[-1]]
            sub_isthing = [torch.tensor([mask_info['isthing'] for mask_info in mask_infos])]
            sub_plural = [torch.tensor([mask_info['plural'] for mask_info in mask_infos])]
            pixel_acc = [torch.eq(pred_masks, gt_masks).float().flatten(1, 2).mean(-1)]

            mask_ious += sub_mask_ious
            isthing += sub_isthing
            plural += sub_plural
            pixel_accs += pixel_acc

        mask_ious = gather_object(mask_ious)
        isthing = gather_object(isthing)
        plural = gather_object(plural)
        pixel_accs = gather_object(pixel_accs)

    if accelerator.is_main_process:
        mask_ious = torch.cat(mask_ious)
        isthing = torch.cat(isthing)
        plural = torch.cat(plural)

        AA = average_accuracy(mask_ious)
        AA_singulars = average_accuracy(mask_ious[torch.logical_not(plural)])
        AA_plurals = average_accuracy(mask_ious[plural])
        AA_things = average_accuracy(mask_ious[isthing])
        AA_stuff = average_accuracy(mask_ious[torch.logical_not(isthing)])

        accuracy = (mask_ious > 0.5).float().mean()

        pixel_accs = torch.cat(pixel_accs).mean()

        print(f"aIoU: {AA}, aIoU_singulars: {AA_singulars}, aIoU_plurals: {AA_plurals}, "
              f"aIoU_things: {AA_things}, aIoU_stuff: {AA_stuff}, aAcc@0.5: {accuracy}, "
              f"pixel_accs: {pixel_accs}", flush=True)
