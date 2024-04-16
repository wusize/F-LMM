import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel

@torch.no_grad()
def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection / (union + 1e-12)


class FrozenFuyu(BaseModel):

    def __init__(self,
                 model,
                 tokenizer,
                 mask_head,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None):
        super().__init__()
        self.fuyu = BUILDER.build(model)
        self.fuyu.requires_grad_(False)
        self.tokenizer = BUILDER.build(tokenizer)
        in_channels = self.fuyu.config.num_attention_heads * self.fuyu.config.num_hidden_layers
        mask_head.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.patch_size = self.fuyu.config.patch_size
        self.image_placeholder_id = self.tokenizer.vocab['|SPEAKER|']
        self.image_newline_id = self.tokenizer.vocab['|NEWLINE|']

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
        self.fuyu.train(mode=False)
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

    def _forward(self, data_sample):
        input_ids = data_sample['input_ids'].to(self.fuyu.device)
        mask_ids = data_sample['mask_ids'].to(self.fuyu.device)
        image_tensor = data_sample['pixel_values'].to(device=self.fuyu.device, dtype=self.fuyu.dtype)

        image_patches, image_patches_indices, image_token_ids = self._patchify(image_tensor[None])

        mask_ids = torch.cat([-torch.ones_like(image_token_ids[0]), mask_ids], dim=0)   # len
        image_patches_indices = torch.cat([image_patches_indices,
                                           -torch.ones_like(input_ids)[None]], dim=-1)
        input_ids = torch.cat([image_token_ids, input_ids[None]], dim=1)   # bs=1, len
        attention_mask = torch.ones_like(input_ids)

        meta_data = data_sample['meta_data']
        with torch.no_grad():
            outputs = self.fuyu(input_ids=input_ids,
                                image_patches=image_patches,
                                image_patches_indices=image_patches_indices,
                                attention_mask=attention_mask,
                                # output_hidden_states=True,
                                output_attentions=True)

        attentions = [attn[0, ..., image_patches_indices[0] >= 0]
                      for attn in outputs.attentions]
        del outputs

        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        fuyu_h, fuyu_w = padded_h // self.patch_size,  padded_w // self.patch_size

        attentions = [attn.view(*attn.shape[:-1], fuyu_h, fuyu_w) for attn in attentions]
        masks = data_sample['masks']
        mask_attentions = []

        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]))

        del attentions
        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)
        if self.training:
            mask_attentions.requires_grad = True
        pred_masks = self.mask_head(mask_attentions)[:, 0]
        # todo: unpad pred_masks
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]

        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)

        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)

        pred_masks = pred_masks[:, before_height:before_height+mask_h, before_width:before_width+mask_w]

        return pred_masks


    def compute_loss(self, data):
        mask_cnts = 0
        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0

        for data_sample in data:
            pred_masks = self._forward(data_sample)
            masks = data_sample['masks'].to(self.fuyu.device)
            gt_masks = F.interpolate(masks[None].float(),
                                     size=pred_masks.shape[-2:])[0].to(pred_masks)
            mask_cnt = pred_masks.shape[0]
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
        pred_masks = self._forward(data_sample)

        return pred_masks

    def _patchify(self, inputs):
        bs, c, h, w = inputs.shape
        image_patches = inputs.view(bs, c, h // self.patch_size, self.patch_size,
                                    w // self.patch_size, self.patch_size)
        image_patches = image_patches.permute(
            0, 2, 4, 3, 5, 1).contiguous().view(bs, -1, (self.patch_size ** 2) * c)

        image_patches_indices = torch.arange(
            image_patches.shape[1], dtype=torch.long,
            device=self.fuyu.device).view(h // self.patch_size, w // self.patch_size)
        image_patches_indices = torch.cat([image_patches_indices,
                                           -torch.ones_like(image_patches_indices[:, :1])
                                           ], dim=-1)

        image_token_ids = torch.ones_like(image_patches_indices) * self.image_placeholder_id
        image_token_ids[:, -1] = self.image_newline_id

        image_patches_indices = image_patches_indices.view(1, -1).repeat(bs, 1)
        image_token_ids = image_token_ids.view(1, -1).repeat(bs, 1)

        return image_patches.to(self.fuyu.dtype), image_patches_indices, image_token_ids


from time import time
class FrozenFuyuSAM(FrozenFuyu):
    def __init__(self,
                 sam,
                 sam_weight=1.0,
                 intermediate_weight=1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        self.text_proj = nn.Linear(self.fuyu.config.hidden_size,
                                   self.sam.model.prompt_encoder.embed_dim)
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.fuyu.config.num_hidden_layers))
        self.sam_weight = sam_weight
        self.intermediate_weight = intermediate_weight

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def _forward(self, data_sample):
        # import pdb; pdb.set_trace()
        text_layer_weights = self.get_text_layer_weights()
        input_ids = data_sample['input_ids'].to(self.fuyu.device)
        mask_ids = data_sample['mask_ids'].to(self.fuyu.device)
        image_tensor = data_sample['pixel_values'].to(device=self.fuyu.device, dtype=self.fuyu.dtype)

        image_patches, image_patches_indices, image_token_ids = self._patchify(image_tensor[None])

        mask_ids = torch.cat([-torch.ones_like(image_token_ids[0]), mask_ids], dim=0)   # len
        image_patches_indices = torch.cat([image_patches_indices,
                                           -torch.ones_like(input_ids)[None]], dim=-1)
        input_ids = torch.cat([image_token_ids, input_ids[None]], dim=1)   # bs=1, len
        attention_mask = torch.ones_like(input_ids)

        meta_data = data_sample['meta_data']
        tik = time()
        with torch.no_grad():
            outputs = self.fuyu(input_ids=input_ids,
                                image_patches=image_patches,
                                image_patches_indices=image_patches_indices,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                output_attentions=True)


        print(f"Fuyu forward time: {time() - tik}", flush=True)

        attentions = [attn[0, ..., image_patches_indices[0] >= 0]
                      for attn in outputs.attentions]
        hidden_states = outputs.hidden_states[-self.fuyu.config.num_hidden_layers:]

        del outputs

        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        fuyu_h, fuyu_w = padded_h // self.patch_size,  padded_w // self.patch_size

        attentions = [attn.view(*attn.shape[:-1], fuyu_h, fuyu_w) for attn in attentions]
        masks = data_sample['masks']
        mask_attentions = []
        text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]))


            # num_layers, matched_seq_len, hidden_size
            matched_hidden_states = torch.stack([hs[0, matched] for hs in hidden_states])
            matched_hidden_states *= text_layer_weights.view(-1, 1, 1)
            # matched_seq_len, hidden_size
            text_embeds.append(self.text_proj(matched_hidden_states.sum(0).to(self.sam.dtype)))
        del attentions, hidden_states

        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)
        if self.training:
            mask_attentions.requires_grad = True
        tik = time()
        pred_masks = self.mask_head(mask_attentions)[:, 0]
        print(f"Mask head forward time: {time() - tik}", flush=True)
        # todo: unpad pred_masks
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]

        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)

        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)

        pred_masks = pred_masks[:, before_height:before_height+mask_h, before_width:before_width+mask_w].contiguous()

        tik = time()
        sam_pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)
        print(f"SAM forward time: {time() - tik}", flush=True)

        return pred_masks, sam_pred_masks


    @torch.no_grad()
    def predict(self, data_sample):
        _, pred_masks = self._forward(data_sample)

        return pred_masks

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

        for data_sample in data:
            pred_masks, sam_pred_masks = self._forward(data_sample)
            masks = data_sample['masks'].to(self.fuyu.device)
            gt_masks = F.interpolate(masks[None].float(),
                                     size=pred_masks.shape[-2:])[0].to(pred_masks)
            sam_gt_masks = F.interpolate(masks[None].float(),
                                         size=sam_pred_masks.shape[-2:])[0].to(sam_pred_masks)

            mask_cnt = pred_masks.shape[0]
            assert pred_masks.shape == gt_masks.shape
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
