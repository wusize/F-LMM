import torch
import torch.nn as nn
import torch.nn.functional as F
from frozen_llava.models.meta_arch import FrozenLlava, compute_mask_IoU
from frozen_llava.models.segment_modules.sam_wrapper import SAMWrapper


class FrozenLlavaSAM(FrozenLlava):
    def __init__(self,
                 sam_name, sam_checkpoint,
                 sam_weight=1.0,
                 intermediate_weight=0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = SAMWrapper(model_name=sam_name, checkpoint=sam_checkpoint)
        self.text_proj = nn.Linear(self.llava.config.vision_config.hidden_size,
                                   self.sam.model.prompt_encoder.embed_dim)
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llava.config.text_config.num_hidden_layers))
        self.sam_weight = sam_weight
        self.intermediate_weight = intermediate_weight

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def compute_loss(self, data):
        mask_cnts = 0
        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0
        sam_loss_dice = 0
        sam_loss_mask = 0
        sam_accuracy = 0
        sam_aiou = 0
        text_layer_weights = self.get_text_layer_weights()
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
                                     attention_mask=attention_mask,
                                     output_hidden_states=True,
                                     output_attentions=True)
            fine_image_feature_h, fine_image_feature_w = outputs['image_feature_shapes'][0]
            mask_ids = outputs['mask_ids']
            attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                          for attn in outputs.attentions]
            hidden_states = outputs.hidden_states
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
            text_embeds = []
            for mask_id in range(len(masks)):
                matched = mask_ids[0] == mask_id
                assert matched.sum() > 0

                mask_attentions_with_coarse = torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions_with_coarse])
                mask_attentions_with_fine = torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions_with_fine])
                attentions_with_coarse_list.append(mask_attentions_with_coarse)
                attentions_with_fine_list.append(mask_attentions_with_fine)

                # num_layers, matched_seq_len, hidden_size
                matched_hidden_states = torch.stack([hs[0, matched] for hs in hidden_states])
                matched_hidden_states *= text_layer_weights.view(-1, 1, 1)
                # matched_seq_len, hidden_size
                text_embeds.append(self.text_proj(matched_hidden_states.sum(0)))

            del hidden_states

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
            pred_masks = self.mask_head(attention_maps)[:, 0]
            gt_masks = F.interpolate(masks.to(attention_maps)[None].float(),
                                     size=(fine_image_feature_h, fine_image_feature_w))[0].to(self.llava.dtype)
            assert pred_masks.shape == gt_masks.shape
            import pdb; pdb.set_trace()
            sam_pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)
            sam_gt_masks = F.interpolate(masks.to(attention_maps)[None].float(),
                                         size=sam_pred_masks.shape[-2:])[0].to(self.llava.dtype)
            mask_cnts += mask_cnt

            loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(pred_masks, gt_masks)
            loss_dice += loss_dice_ * mask_cnt
            loss_mask += loss_mask_ * mask_cnt
            accuracy += accuracy_ * mask_cnt
            aiou += aiou_ * mask_cnt

            sam_loss_dice_, sam_loss_mask_, sam_accuracy_, sam_aiou_ = self._compute(sam_pred_masks, sam_gt_masks)
            sam_loss_dice += sam_loss_dice_ * mask_cnt
            sam_loss_mask += sam_loss_mask_ * mask_cnt
            sam_accuracy += sam_accuracy_ * mask_cnt
            sam_aiou += sam_aiou_ * mask_cnt

        assert mask_cnts > 0
        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts,
                     'sam_loss_mask': sam_loss_mask / mask_cnts,
                     'sam_loss_dice': sam_loss_dice / mask_cnts,
                     'sam_accuracy': sam_accuracy / mask_cnts,
                     'sam_aiou': sam_aiou / mask_cnts,
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
