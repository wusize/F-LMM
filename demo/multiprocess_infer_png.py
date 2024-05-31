import os
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from src.datasets.png import PNGDataset
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN
from demo.utils import colors

def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)

def do_kmeans(feature_map, gt_mask):
    c, h, w = feature_map.shape
    feature_map = feature_map.view(c, h*w).T.contiguous()
    feature_map = F.normalize(feature_map, dim=-1).cpu().numpy()
    cluster_method = KMeans(n_clusters=2, n_init=10)
    # fit model and predict clusters
    results = cluster_method.fit_predict(feature_map)

    mask1 = torch.from_numpy(results.reshape(h, w) == 0).float()
    mask2 = torch.from_numpy(results.reshape(h, w) == 1).float()

    masks = F.interpolate(torch.stack([mask1, mask2])[None], size=gt_mask.shape, mode='bilinear')[0]
    ious = compute_mask_IoU(masks.view(2, -1), torch.from_numpy(gt_mask).float().view(1, -1))[-1]

    return masks[ious.argmax()] > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint',
                        default='checkpoints/frozen_deepseek_vl_1_3b_unet_sam_l_iter_95080.pth', type=str)
    parser.add_argument('--save_dir', default='data/deepseek1_3b_png', type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    for subset in ['gt', 'sam', 'conv', 'attn', 'attn_all']:
        os.makedirs(os.path.join(args.save_dir, subset), exist_ok=True)

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

    print(f'Device: {accelerator.device}', flush=True)
    model = BUILDER.build(cfg.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model.eval()

    dataset_params = dict(json_file='data/png_coco_val2017.json',
                          panoptic_json_file='data/coco/annotations/panoptic_val2017.json',
                          panoptic_png_path='data/coco/panoptic_val2017',
                          tokenizer=tokenizer,
                          image_processor=image_processor,
                          prompt_template=prompt_template,
                          local_path='data/coco/val2017',
                          ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/val2017',
                          image2tensor=cfg.get('image2tensor', True),
                          add_image_token=cfg.get('add_image_token', False),
                          image_token=cfg.get('image_token', DEFAULT_IMAGE_TOKEN)
    )
    if prompt is not None:
        dataset_params.update(prompt=prompt)
    png_dataset = PNGDataset(**dataset_params)

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
            with torch.no_grad():
                output = model._forward(data_sample)
            gt_masks = data_sample['gt_masks'].cpu().numpy() > 0
            pred_masks = F.interpolate(output['pred_masks'][None].float().cpu(),
                                       size=gt_masks.shape[-2:], mode='bilinear')[0].numpy() > 0
            sam_pred_masks = F.interpolate(output['sam_pred_masks'][None].float().cpu(),
                                           size=gt_masks.shape[-2:], mode='bilinear')[0].numpy() > 0
            mask_attentions = output['mask_attentions']
            attn_masks = torch.stack([do_kmeans(mask_attention, gt_mask) for mask_attention, gt_mask in
                                      zip(mask_attentions, gt_masks)])
            # attn_masks = F.interpolate(attn_masks[None].float().cpu(),
            #                            size=gt_masks.shape[-2:], mode='bilinear')[0].numpy() > 0

            file_name = os.path.basename(data_sample['file_name'])

            image = np.array(data_sample['image']).astype(np.float32)
            sam_image = image.copy()
            gt_image = image.copy()
            conv_image = image.copy()
            attn_image = image.copy()

            for color_id, (gt_mask, sam_mask, cnn_mask, attn_mask) in enumerate(
                    zip(gt_masks, sam_pred_masks, pred_masks, attn_masks)):
                sam_image[sam_mask] = sam_image[sam_mask] * 0.2 + np.array(colors[color_id]).reshape((1, 1, 3)) * 0.8
                gt_image[gt_mask] = gt_image[gt_mask] * 0.2 + np.array(colors[color_id]).reshape((1, 1, 3)) * 0.8
                conv_image[cnn_mask] = conv_image[cnn_mask] * 0.2 + np.array(colors[color_id]).reshape((1, 1, 3)) * 0.8
                attn_image[attn_mask] = attn_image[attn_mask] * 0.2 + np.array(colors[color_id]).reshape((1, 1, 3)) * 0.8


            all_in_one = np.concatenate([image, attn_image, conv_image, sam_image, gt_image], axis=1)

            sam_image = Image.fromarray(sam_image.astype(np.uint8))
            gt_image = Image.fromarray(gt_image.astype(np.uint8))
            conv_image = Image.fromarray(conv_image.astype(np.uint8))
            attn_image = Image.fromarray(attn_image.astype(np.uint8))

            all_in_one = Image.fromarray(all_in_one.astype(np.uint8))


            sam_image.save(os.path.join(args.save_dir, f'sam/{file_name}'))
            gt_image.save(os.path.join(args.save_dir, f'gt/{file_name}'))
            conv_image.save(os.path.join(args.save_dir, f'conv/{file_name}'))
            attn_image.save(os.path.join(args.save_dir, f'attn/{file_name}'))
            all_in_one.save(os.path.join(args.save_dir, file_name))

            np.save(os.path.join(args.save_dir, f'attn_all/{file_name[:-4]}.npy'), mask_attentions.cpu().numpy())
