import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry, predictor, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide


def mask2box(mask, original_image_size):
    h, w = mask.shape
    original_h, original_w = original_image_size
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    return np.array([x0, y0, x1, y1]) * np.array([original_w/w, original_h/h] * 2)


class SAMWrapper(nn.Module):
    def __init__(self, model_name, checkpoint):
        super(SAMWrapper, self).__init__()
        self.model = sam_model_registry[model_name](checkpoint=checkpoint)
        self.model.image_encoder.requires_grad_(False)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def train(self, mode=True):
        super().train(mode=mode)
        self.model.image_encoder.eval()
        self.training = mode
        return self

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

    def forward(self, image, pred_masks, text_embeds):
        # masks are in logits
        import pdb; pdb.set_trace()
        image_embedding, original_image_size, input_size = self.encode_image(image)
        image_embedding.requires_grad = True
        prompt_masks = F.interpolate(pred_masks[:, None].float(), size=(256, 256), mode='bilinear').to(pred_masks)

        sam_masks = []
        for prompt_mask, pred_mask, text_embed in zip(prompt_masks, pred_masks, text_embeds):
            box = mask2box(pred_mask.detach().cpu().numpy(), original_image_size)
            box = self.transform.apply_boxes(box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=pred_mask.dtype, device=self.model.device)
            box_torch = box_torch[None, :]    # 1, 1, 4
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=prompt_mask.view(1, 1, 256, 256),
            )
            sparse_embeddings = torch.cat([sparse_embeddings, text_embed[None]], dim=1)
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            sam_mask = self.model.postprocess_masks(low_res_masks, input_size, original_image_size)
            sam_masks.append(sam_mask[0, 0])
            
        return torch.stack(sam_masks)
