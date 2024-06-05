import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def mask2box(mask):
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    return np.array([x0, y0, x1+1, y1+1])    # avoid x0==x1


def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)


class SAMWrapper(nn.Module):
    def __init__(self, model_name, checkpoint,
                 use_text=True, use_mask=True, use_box=True,
                 multimask_output=False):
        super(SAMWrapper, self).__init__()
        self.model = sam_model_registry[model_name](checkpoint=checkpoint)
        self.model.image_encoder.requires_grad_(False)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.use_text = use_text
        self.use_mask = use_mask
        self.use_box = use_box
        self.multimask_output = multimask_output

    def train(self, mode=True):
        super().train(mode=mode)
        self.model.image_encoder.eval()
        self.training = mode
        return self

    @property
    def dtype(self):
        return self.model.dtype

    @torch.no_grad()
    def encode_image(self, image):
        image = np.array(image.convert(self.model.image_format))
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.model.device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        original_image_size = image.shape[:2]
        input_size = transformed_image.shape[-2:]

        features = self.model.image_encoder(self.model.preprocess(transformed_image))

        return features, original_image_size, input_size

    def generate_prompt_masks(self, masks, input_size):
        pad_value = min(-1.0, masks.min().item())
        masks = F.interpolate(masks[:, None].float(), size=input_size, mode='bilinear').to(masks)
        h, w = masks.shape[-2:]
        masks = F.pad(masks, (0, self.model.image_encoder.img_size - w,
                              0, self.model.image_encoder.img_size - h), value=pad_value)
        prompt_masks = F.interpolate(masks.float(), size=(256, 256), mode='bilinear').to(masks)

        return prompt_masks

    def forward(self, image, pred_masks, text_embeds):
        # masks are in logits
        image_embedding, original_image_size, input_size = self.encode_image(image)
        if self.training:
            image_embedding.requires_grad = True
        prompt_masks = self.generate_prompt_masks(pred_masks, input_size)

        pred_masks = F.interpolate(pred_masks.detach()[None].float().sigmoid(),
                                   size=original_image_size, mode='bilinear')[0]
        pred_masks = (pred_masks > 0.5).to(pred_masks)

        sam_masks = []
        for prompt_mask, pred_mask, text_embed in zip(prompt_masks, pred_masks, text_embeds):
            if self.use_box:
                if pred_mask.sum() > 0:
                    box = mask2box(pred_mask.float().cpu().numpy())
                else:
                    h, w = original_image_size
                    box = np.array([0.0, 0.0, w, h])
                box = self.transform.apply_boxes(box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=pred_mask.dtype, device=self.model.device)
                box_torch = box_torch[None, :]    # 1, 1, 4
            else:
                box_torch = None
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=prompt_mask.view(1, 1, 256, 256) if self.use_mask else None,
            )
            if self.use_text:
                sparse_embeddings = torch.cat([sparse_embeddings.to(dense_embeddings),
                                               text_embed[None].to(dense_embeddings)], dim=1)
            else:
                sparse_embeddings = sparse_embeddings.to(dense_embeddings)
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output,
            )
            sam_mask = self.model.postprocess_masks(low_res_masks, input_size, original_image_size)

            if self.multimask_output:
                candidate_masks = (sam_mask[0] > 0.0).float()
                candidate_ious = compute_mask_IoU(candidate_masks.view(3, -1),
                                                  pred_mask.float().view(1, -1))[-1]
                sam_mask = sam_mask[0, candidate_ious.argmax()]
            else:
                assert sam_mask.shape[1] == 1
                sam_mask = sam_mask[0, 0]
            sam_masks.append(sam_mask)

        return torch.stack(sam_masks)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if 'image_encoder' not in k}
