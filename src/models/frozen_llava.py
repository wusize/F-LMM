import torch
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel


@torch.no_grad()
def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection / (union + 1e-12)


class FrozenLlavaNext(BaseModel):

    def __init__(self,
                 model,
                 mask_head,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None):
        super().__init__()
        self.llava = BUILDER.build(model)
        self.llava.requires_grad_(False)
        in_channels = (self.llava.config.text_config.num_attention_heads *
                       self.llava.config.text_config.num_hidden_layers*2)
        mask_head.update(
            in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.patch_size = self.llava.config.vision_config.patch_size
        self.merge = merge
        assert merge in ['mean', 'max']

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

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

    def compute_loss(self, data):
        mask_cnts = 0
        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0

        for data_sample in data:
            inputs = dict(input_ids=data_sample['input_ids'][None].to(self.llava.device),
                          mask_ids=data_sample['mask_ids'][None].to(self.llava.device),
                          pixel_values=data_sample['pixel_values'][None].to(device=self.llava.device,
                                                                            dtype=self.llava.dtype)
                          )
            attention_mask = torch.ones(inputs['input_ids'].shape, device=self.llava.device,
                                        dtype=torch.bool)
            meta_data = data_sample['meta_data']
            with torch.no_grad():
                outputs = self.llava(**inputs,
                                     attention_mask=attention_mask,
                                     # output_hidden_states=True,
                                     output_attentions=True)
            mask_ids = outputs['mask_ids']
            attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                          for attn in outputs.attentions]
            del outputs

            llava_h, llava_w = (meta_data['padded_shape']['height'] // self.patch_size,
                                meta_data['padded_shape']['width'] // self.patch_size)

            attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
            masks = data_sample['masks'].to(self.llava.device)
            mask_attentions = []

            for mask_id in range(len(masks)):
                matched = mask_ids[0] == mask_id
                assert matched.sum() > 0
                mask_attentions.append(torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]))

            del attentions
            mask_attentions = torch.stack(mask_attentions)
            mask_attentions.requires_grad = True
            mask_cnt = mask_attentions.shape[0]
            pred_masks = self.mask_head(mask_attentions)[:, 0]
            gt_masks = F.interpolate(masks.to(mask_attentions)[None].float(),
                                     size=pred_masks.shape[-2:])[0].to(self.llava.dtype)
            assert pred_masks.shape == gt_masks.shape
            mask_cnts += mask_cnt

            loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(pred_masks, gt_masks)
            loss_dice += loss_dice_ * mask_cnt
            loss_mask += loss_mask_ * mask_cnt
            accuracy += accuracy_ * mask_cnt
            aiou += aiou_ * mask_cnt


        assert mask_cnts > 0
        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts,
                     }
        for k, v in loss_dict.items():
            if 'sam_loss' in k:
                loss_dict[k] *= self.sam_weight
            elif 'loss' in k:
                loss_dict[k] *= self.intermediate_weight

        return loss_dict

    def _compute(self, pred_masks, gt_masks):
        mask_cnt = pred_masks.shape[0]
        loss_dice = self.loss_dice(
            pred_masks.view(mask_cnt, -1), gt_masks.view(mask_cnt, -1),
            avg_factor=mask_cnt)
        loss_mask = self.loss_mask(
            pred_masks.view(-1),
            gt_masks.view(-1),
            avg_factor=pred_masks.numel())
        accuracy = torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks),
                            gt_masks).to(gt_masks).mean()
        aiou = compute_mask_IoU((pred_masks.detach().sigmoid() > 0.5).to(gt_masks).view(mask_cnt, -1),
                                gt_masks.view(mask_cnt, -1)).mean()

        return loss_dice, loss_mask, accuracy, aiou

    @torch.no_grad()
    def predict(self, data_sample):
        inputs = dict(input_ids=data_sample['input_ids'][None].to(self.llava.device),
                      mask_ids=data_sample['mask_ids'][None].to(self.llava.device),
                      pixel_values=data_sample['pixel_values'][None].to(device=self.llava.device,
                                                                        dtype=self.llava.dtype)
                      )
        attention_mask = torch.ones(inputs['input_ids'].shape, device=self.llava.device,
                                    dtype=torch.bool)
        meta_data = data_sample['meta_data']
        with torch.no_grad():
            outputs = self.llava(**inputs,
                                 attention_mask=attention_mask,
                                 # output_hidden_states=True,
                                 output_attentions=True)
        mask_ids = outputs['mask_ids']
        attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                      for attn in outputs.attentions]
        del outputs
        llava_h, llava_w = (meta_data['padded_shape']['height'] // self.patch_size,
                            meta_data['padded_shape']['width'] // self.patch_size)

        attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
        masks = data_sample['masks'].to(self.llava.device)
        mask_attentions = []

        for mask_id in range(len(masks)):
            matched = mask_ids[0] == mask_id
            assert matched.sum() > 0
            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]))
        del attentions
        mask_attentions = torch.stack(mask_attentions)
        pred_masks = self.mask_head(mask_attentions)[:, 0]

        return pred_masks
