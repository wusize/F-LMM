import torch
import torch.nn.functional as F
import argparse
import numpy as np
from mmengine.config import Config
from xtuner.registry import BUILDER
from sklearn.cluster import KMeans
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)


def do_kmeans(feature_map, ref_mask):
    c, h, w = feature_map.shape
    feature_map = feature_map.view(c, h*w).T.contiguous()
    feature_map = F.normalize(feature_map, dim=-1).cpu().numpy()
    cluster_method = KMeans(n_clusters=2, n_init=10)
    # fit model and predict clusters
    results = cluster_method.fit_predict(feature_map)

    mask1 = torch.from_numpy(results.reshape(h, w) == 0).float()
    mask2 = torch.from_numpy(results.reshape(h, w) == 1).float()

    masks = F.interpolate(torch.stack([mask1, mask2])[None], size=ref_mask.shape, mode='bilinear')[0]
    ious = compute_mask_IoU(masks.view(2, -1), ref_mask.float().view(1, -1))[-1]

    return masks[ious.argmax()] > 0


def draw_mask(image, mask):
    image = np.array(image.convert('RGB')).astype(np.float32)
    image[mask] = image[mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32).reshape(1, 1, 3) * 0.5
    image = image.astype(np.uint8)

    return Image.fromarray(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--image', default='images/example.jpg', type=str)
    parser.add_argument('--checkpoint',
                        default='checkpoints/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.pth', type=str)
    parser.add_argument('--text', default='The cat left to the dog.', type=str)
    parser.add_argument('--output', default='output.jpg', type=str)

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template

    model = BUILDER.build(cfg.model)
    image_processor = BUILDER.build(cfg.image_processor)
    tokenizer = BUILDER.build(cfg.tokenizer)
    state_dict = guess_load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)

    image_placeholder = cfg.get('image_placeholder', "<image>\n")
    model = model.cuda()
    model.eval()

    image2tensor = cfg.get('image2tensor', True)
    add_image_token = cfg.get('add_image_token', False)
    image_token = cfg.get('image_token', DEFAULT_IMAGE_TOKEN)

    if add_image_token:
        print(f"Manually add image token: {image_token}")
        special_tokens_dict = {'additional_special_tokens': [image_token, ]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 1

    image_token_idx = tokenizer.encode(image_token, add_special_tokens=False)[-1]
    print(f"Image token: {tokenizer.decode(image_token_idx)}")

    prompt_tokens = tokenizer.encode(
        prompt_template['INSTRUCTION'].format(input=image_placeholder),
        add_special_tokens=True, return_tensors='pt')[0]
    object_tokens = tokenizer.encode(
        args.text, add_special_tokens=False, return_tensors='pt')[0]

    input_ids = torch.cat([prompt_tokens, object_tokens]).cuda()
    image = Image.open(args.image)

    image_data = image_processor.preprocess(image)

    pixel_values = image_data['pixel_values'][0]
    if image2tensor:
        pixel_values = torch.from_numpy(pixel_values)
    meta_data = image_data['meta_datas'][0]

    if add_image_token:
        input_ids[input_ids == image_token_idx] = IMAGE_TOKEN_INDEX

    outputs = model.forward_lmm(dict(pixel_values=pixel_values,
                                     input_ids=input_ids))

    hidden_states = outputs['hidden_states'][-len(object_tokens):]
    attentions = outputs['attentions'][:, -len(object_tokens):]
    import pdb; pdb.set_trace()

    mask_attentions, cnn_pred_masks, sam_pred_masks = model.forward_seg(
        attentions, hidden_states, dict(image=image, meta_data=meta_data))
    attn_mask = do_kmeans(mask_attentions.float(), sam_pred_masks.detach().cpu() > 0.0)
    draw_mask(image, sam_pred_masks.detach().cpu().numpy() > 0).save(args.output)
    draw_mask(image, attn_mask.numpy()).save(args.output.replace('.jpg', '_attn.jpg'))
    draw_mask(image, cnn_pred_masks.detach().cpu().numpy() > 0).save(args.output.replace('.jpg', '_cnn.jpg'))
