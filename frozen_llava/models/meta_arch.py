import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from xtuner.registry import BUILDER

@torch.no_grad()
def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection / (union + 1e-12)


class FrozenLlava(BaseModel):

    def __init__(self,
                 model,
                 mask_head,
                 use_image_hidden_states=False,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None):
        super().__init__()
        self.llava = BUILDER.build(model)
        self.llava.requires_grad_(False)
        in_channels = (self.llava.config.text_config.num_attention_heads *
                       self.llava.config.text_config.num_hidden_layers*2)
        if use_image_hidden_states:
            in_channels += self.llava.config.vision_config.hidden_size
        mask_head.update(
            in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.patch_size = self.llava.config.vision_config.patch_size
        self.merge = merge
        assert merge in ['mean', 'max']

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        self.use_image_hidden_states = use_image_hidden_states

    def apply_merge(self, x, dim=1):
        if self.merge == 'mean':
            return x.mean(dim=dim)
        elif self.merge == 'max':
            return x.max(dim=dim).values
        else:
            raise NotImplementedError

    def init_weights(self):
        pass

    def train(self, mode=True):
        self.llava.train(mode=False)
        self.mask_head.train(mode=mode)
        self.training = mode
        return self

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data)
        elif mode == 'predict':
            return self.predict(data)
        elif mode == 'tensor':
            return self._forward(data)
        else:
            raise NotImplementedError

    def _forward(self, data):
        return None

    def predict(self, data):
        return None

    def compute_loss(self, data):
        mask_cnts = 0
        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0
        for data_sample in data:
            assert data_sample['pixel_values'].shape[0] > 1
            inputs = dict(input_ids=data_sample['input_ids'][None].to(self.llava.device),
                          mask_ids=data_sample['mask_ids'][None].to(self.llava.device),
                          pixel_values=data_sample['pixel_values'][None].to(device=self.llava.device,
                                                                            dtype=self.llava.dtype),
                          image_sizes=data_sample['image_sizes'][None].to(self.llava.device))
            attention_mask = torch.ones(inputs['input_ids'].shape, device=self.llava.device,
                                        dtype=torch.bool)
            with torch.no_grad():
                outputs = self.llava(**inputs,
                                     attention_mask=attention_mask, output_attentions=True)
            fine_image_feature_h, fine_image_feature_w = outputs['image_feature_shapes'][0]
            mask_ids = outputs['mask_ids']
            attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                          for attn in outputs.attentions]
            image_hidden_states = outputs.image_hidden_states[0]
            del outputs

            coarse_image_h, coarse_image_w = data_sample['pixel_values'].shape[2:]
            coarse_image_feature_h, coarse_image_feature_w = (
                coarse_image_h // self.patch_size, coarse_image_w // self.patch_size)

            attentions_with_coarse = [
                attn[..., :coarse_image_feature_h*coarse_image_feature_w].view(
                    *attn.shape[:-1], coarse_image_feature_h, coarse_image_feature_w
                ) for attn in attentions]
            attentions_with_fine = [
                attn[..., coarse_image_feature_h*coarse_image_feature_w:].view(
                    *attn.shape[:-1], fine_image_feature_h, fine_image_feature_w+1
                )[..., :-1] for attn in attentions]
            del attentions
            masks = data_sample['masks'].to(self.llava.device)

            attentions_with_coarse_list = []
            attentions_with_fine_list = []
            for mask_id in range(len(masks)):
                matched = mask_ids[0] == mask_id
                assert matched.sum() > 0

                mask_attentions_with_coarse = torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions_with_coarse])
                mask_attentions_with_fine = torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions_with_fine])
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
            ], dim=1).to(self.llava.dtype)
            del attentions_with_coarse, attentions_with_fine
            attention_maps.requires_grad = True
            # print(f"============={attention_maps.dtype}===========", flush=True)
            mask_cnt = attention_maps.shape[0]
            if self.use_image_hidden_states:
                attention_maps = torch.cat([attention_maps,
                                            image_hidden_states[None].repeat(mask_cnt, 1, 1, 1)],
                                           dim=1)
            del image_hidden_states
            pred_masks = self.mask_head(attention_maps)[:, 0]
            gt_masks = F.interpolate(masks.to(attention_maps)[None].float(),
                                     size=(fine_image_feature_h, fine_image_feature_w))[0].to(self.llava.dtype)
            assert pred_masks.shape == gt_masks.shape
            mask_cnts += mask_cnt

            # dice loss
            loss_dice += self.loss_dice(
                pred_masks.view(mask_cnt, -1), gt_masks.view(mask_cnt, -1),
                avg_factor=mask_cnt) * mask_cnt

            weight = torch.ones_like(pred_masks.view(-1))
            num_pos = gt_masks.sum()
            num_neg = mask_cnt*fine_image_feature_h*fine_image_feature_w - num_pos
            weight[gt_masks.view(-1) > 0] /= (num_pos + 1e-12)
            weight[gt_masks.view(-1) < 1] /= (num_neg + 1e-12)
            # mask loss
            loss_mask += self.loss_mask(
                pred_masks.view(-1),
                gt_masks.view(-1),
                weight=weight,
                avg_factor=2) * mask_cnt
            acc = torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks),
                           gt_masks).to(gt_masks).mean()
            accuracy += acc * mask_cnt
            aiou += compute_mask_IoU((pred_masks.detach().sigmoid() > 0.5).to(gt_masks).view(mask_cnt, -1),
                                     gt_masks.view(mask_cnt, -1)).mean() * mask_cnt

        assert mask_cnts > 0
        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts}
        return loss_dict
