from mmdet.datasets import RefCocoDataset
from flmm.datasets.transforms import PILLoadImageFromFile, RefCOCO2PNG
from mmdet.datasets.transforms import LoadAnnotations
from mmdet.evaluation import RefSegMetric
import argparse
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN
from accelerate import Accelerator
from accelerate.utils import gather_object
from mmdet.structures.mask import BitmapMasks

from tqdm import tqdm
import torch
import torch.nn.functional as F
from time import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ceph', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()

    ### Initialize accelerator
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
    image_processor = cfg.image_processor
    prompt = cfg.get('prompt', None)
    model = BUILDER.build(cfg.model)

    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")

    print(f"Start moving model to device: {accelerator.device}", flush=True)
    tik = time()
    model = model.to(device=accelerator.device)
    print(f"Finished moving model to device: {accelerator.device}, time used: {time() - tik}", flush=True)
    model.eval()

    if args.ceph:
        backend_args = dict(
            backend='petrel',
            path_mapping=dict({
                'data/coco/train2014/': 'openmmlab:s3://openmmlab/datasets/detection/coco/train2014/'
            }))
    else:
        backend_args = None

    refcoco2png_params = dict(
        type=RefCOCO2PNG,
        image_processor=image_processor,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        concat=args.concat,
        image2tensor=cfg.get('image2tensor', True),
        add_image_token=cfg.get('add_image_token', False),
        image_token=cfg.get('image_token', DEFAULT_IMAGE_TOKEN)
    )
    accelerator.print(f"Do concatenation? {args.concat}")
    if prompt is not None:
        refcoco2png_params.update(prompt=prompt)

    # ref_coco data pipeline
    test_pipeline = [
        dict(type=PILLoadImageFromFile, backend_args=backend_args),
        dict(
            type=LoadAnnotations,
            with_mask=True,
            with_bbox=False,
            with_seg=False,
            with_label=False),
        refcoco2png_params
    ]

    refcoco_subsets = dict()
    for split in ['val', 'testA', 'testB']:
        if args.val and split != 'val':
            continue
        refcoco_subsets[f'refcoco_{split}'] = dict(
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split=split)

    for split in ['val', 'testA', 'testB']:
        if args.val and split != 'val':
            continue
        refcoco_subsets[f'refcoco+_{split}'] = dict(
            ann_file='refcoco+/instances.json',
            split_file='refcoco+/refs(unc).p',
            split=split)

    for split in ['val', 'test']:
        if args.val and split != 'val':
            continue
        refcoco_subsets[f'refcocog_{split}'] = dict(
            ann_file='refcocog/instances.json',
            split_file='refcocog/refs(umd).p',
            split=split)

    for name, subset in refcoco_subsets.items():
        accelerator.print(f"Start evaluating {name}")
        dataset = RefCocoDataset(
            data_root='data/coco/',
            data_prefix=dict(img_path='train2014/'),
            text_mode='random' if args.random else 'select_first',
            pipeline=test_pipeline,
            **subset
        )
        # sync GPUs and start the timer
        accelerator.wait_for_everyone()

        data_ids = list(range(len(dataset)))
        if args.debug:
            data_ids = data_ids[:100]

        results = []
        # divide the prompt list onto the available GPUs
        with accelerator.split_between_processes(data_ids) as sub_ids:
            for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
                data_sample = dataset[idx]
                if args.concat:
                    with torch.no_grad():
                        pred_mask_logits = model.predict(data_sample)

                    gt_masks = data_sample['gt_masks'].numpy() > 0
                    pred_masks = F.interpolate(pred_mask_logits[None].float().sigmoid(),
                                               size=gt_masks.shape[-2:], mode='bilinear')[0].cpu()
                    pred_masks = pred_masks > 0.5

                    assert len(pred_masks) == len(gt_masks)
                    mask_cnt = pred_masks.shape[0]

                    # Formulate the output into the format that the evaluator accepts
                    results.append(dict(pred_instances=dict(masks=pred_masks),
                                        gt_masks=BitmapMasks(masks=gt_masks,
                                                             height=gt_masks.shape[1],
                                                             width=gt_masks.shape[2]))
                                   )
                else:
                    for sub_data_sample in data_sample:
                        with torch.no_grad():
                            pred_mask_logits = model.predict(sub_data_sample)

                        gt_masks = sub_data_sample['gt_masks'].numpy() > 0
                        pred_masks = F.interpolate(pred_mask_logits[None].float().sigmoid(),
                                                   size=gt_masks.shape[-2:], mode='bilinear')[0].cpu()
                        pred_masks = pred_masks > 0.5

                        assert len(pred_masks) == len(gt_masks)
                        mask_cnt = pred_masks.shape[0]
                        assert mask_cnt == 1

                        # Formulate the output into the format that the evaluator accepts
                        results.append(dict(pred_instances=dict(masks=pred_masks),
                                            gt_masks=BitmapMasks(masks=gt_masks,
                                                                 height=gt_masks.shape[1],
                                                                 width=gt_masks.shape[2]))
                                       )
            results = gather_object(results)
        if accelerator.is_main_process:
            accelerator.print(f"Collected {len(results)} result samples from all gpus")
            evaluator = RefSegMetric(metric=['cIoU', 'mIoU'])
            evaluator.process(data_batch=dict(), data_samples=results)
            metrics = evaluator.compute_metrics(evaluator.results)
            accelerator.print(f"Evaluation results on {name}: {metrics}")
        accelerator.print(f"Finished evaluating {name}")
