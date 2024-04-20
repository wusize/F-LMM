import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import guess_load_checkpoint


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
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 pretrained=None,
                 key_phrase_head=None):
        super().__init__()
        self.llava = BUILDER.build(model)
        self.llava.requires_grad_(False)
        in_channels = (self.llava.config.text_config.num_attention_heads *
                       self.llava.config.text_config.num_hidden_layers)
        mask_head.update(
            in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.patch_size = self.llava.config.vision_config.patch_size
        self.merge = merge
        assert merge in ['mean', 'max', 'learn']

        if merge == 'learn':
            self.merge_proj = nn.Linear(self.llava.config.text_config.hidden_size,1)

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llava.config.text_config.num_hidden_layers))
        key_phrase_head.update(in_channels=self.llava.config.text_config.hidden_size)
        self.key_phrase_head = BUILDER.build(key_phrase_head)

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)
        
    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def apply_merge(self, x, dim=1, weights=None):
        if self.merge == 'mean':
            return x.mean(dim=dim)
        elif self.merge == 'max':
            return x.max(dim=dim).values
        elif self.merge == 'learn':
            return (x * weights).sum(dim=dim)
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

    def _forward(self, data_sample):
        text_layer_weights = self.get_text_layer_weights()
        inputs = dict(input_ids=data_sample['input_ids'][None].to(self.llava.device),
                      mask_ids=data_sample['mask_ids'][None].to(self.llava.device),
                      pixel_values=data_sample['pixel_values'][None].to(device=self.llava.device,
                                                                        dtype=self.llava.dtype),
                      labels=data_sample['labels'][None].to(self.llava.device)
                      )
        attention_mask = torch.ones(inputs['input_ids'].shape, device=self.llava.device,
                                    dtype=torch.bool)
        meta_data = data_sample['meta_data']
        with torch.no_grad():
            outputs = self.llava(**inputs,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True,
                                 output_attentions=True)
        mask_ids = outputs['mask_ids'][0]
        attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                      for attn in outputs.attentions]
        hidden_states = outputs.hidden_states[-self.llava.config.text_config.num_hidden_layers:]

        labels = outputs.labels[0]

        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        del outputs

        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size

        attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
        masks = data_sample['masks']
        mask_attentions = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            matched_len = matched.sum()
            if self.merge == 'learn':
                merge_weights = self.merge_proj(hidden_states[matched])[:, 0].softmax(0)   # matched_len
                merge_weights = merge_weights.view(1, matched_len, 1, 1)
            else:
                merge_weights = None

            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, matched], dim=1, weights=merge_weights) for attn in attentions]))

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

        pred_masks = pred_masks[:, before_height:before_height+mask_h, before_width:before_width+mask_w].contiguous()

        output = dict(pred_masks=pred_masks,
                      labels=labels, mask_ids=mask_ids, hidden_states=hidden_states)

        return output

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
        return self._forward(data_sample)['pred_masks']


class FrozenLlavaSAM(FrozenLlava):
    def __init__(self, sam, *args, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        self.text_proj = nn.Linear(self.llava.config.text_config.hidden_size,
                                   self.sam.model.prompt_encoder.embed_dim)

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

    def _forward(self, data_sample):
        text_layer_weights = self.get_text_layer_weights()
        inputs = dict(input_ids=data_sample['input_ids'][None].to(self.llava.device),
                      mask_ids=data_sample['mask_ids'][None].to(self.llava.device),
                      pixel_values=data_sample['pixel_values'][None].to(device=self.llava.device,
                                                                        dtype=self.llava.dtype),
                      labels=data_sample['labels'][None].to(self.llava.device)
                      )
        attention_mask = torch.ones(inputs['input_ids'].shape, device=self.llava.device,
                                    dtype=torch.bool)
        meta_data = data_sample['meta_data']
        with torch.no_grad():
            outputs = self.llava(**inputs,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True,
                                 output_attentions=True)
        mask_ids = outputs['mask_ids'][0]
        attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                      for attn in outputs.attentions]
        hidden_states = outputs.hidden_states[-self.llava.config.text_config.num_hidden_layers:]

        labels = outputs.labels[0]

        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        del outputs

        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size

        attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
        masks = data_sample['masks']
        mask_attentions = []
        text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            matched_len = matched.sum()
            if self.merge == 'learn':
                merge_weights = self.merge_proj(hidden_states[matched])[:, 0].softmax(0)   # matched_len
                merge_weights = merge_weights.view(1, matched_len, 1, 1)
            else:
                merge_weights = None

            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, matched], dim=1, weights=merge_weights) for attn in attentions]))
            text_embeds.append(self.text_proj(hidden_states[matched]))

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
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()
        sam_pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)

        output = dict(pred_masks=pred_masks, sam_pred_masks=sam_pred_masks,
                      labels=labels, mask_ids=mask_ids, hidden_states=hidden_states)

        return output

    @torch.no_grad()
    def predict(self, data_sample):
        return self._forward(data_sample)['sam_pred_masks']

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

        # losses_dice_phrase = []
        # losses_mask_phrase = []
        # losses_cls_phrase = []
        # aious_phrase = []

        for data_sample in data:
            forward_output = self._forward(data_sample)
            pred_masks, sam_pred_masks = forward_output['pred_masks'], forward_output['sam_pred_masks']
            masks = data_sample['masks'].to(self.llava.device)
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

            # labels, mask_ids, hidden_states = (forward_output['labels'],
            #                                    forward_output['mask_ids'], forward_output['hidden_states'])
            # loss_dice_phrase, loss_mask_phrase, loss_cls_phrase, aiou_phrase = self.key_phrase_head(
            #     hidden_states[labels >= 0], mask_ids[labels >= 0])
            # losses_dice_phrase.append(loss_dice_phrase)
            # losses_mask_phrase.append(loss_mask_phrase)
            # losses_cls_phrase.append(loss_cls_phrase)
            # aious_phrase.append(aiou_phrase)

        assert mask_cnts > 0

        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts,
                     'sam_loss_mask': sam_loss_mask / mask_cnts,
                     'sam_loss_dice': sam_loss_dice / mask_cnts,
                     'sam_accuracy': sam_accuracy / mask_cnts,
                     'sam_aiou': sam_aiou / mask_cnts,
                     # 'loss_dice_phrase': sum(losses_dice_phrase) / len(data),
                     # 'loss_mask_phrase': sum(losses_mask_phrase) / len(data),
                     # 'loss_cls_phrase': sum(losses_cls_phrase) / len(data),
                     # 'aiou_phrase': sum(aious_phrase) / len(data)
                     }

        return loss_dict

    @torch.no_grad()
    def gcg_forward(self, data_sample, **kwargs):
        # for now we implement greedy search only
        input_ids = data_sample['input_ids'][None].to(self.llava.device)
        pixel_values = data_sample['pixel_values'][None].to(device=self.llava.device,
                                                            dtype=self.llava.dtype)
        attention_mask = torch.ones(input_ids.shape, device=self.llava.device, dtype=torch.bool)
        output = self.llava(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            use_cache=True)
        image_to_overwrite = output.image_to_overwrite[0]
        past_key_values = output.past_key_values
        past_length = past_key_values[0][0].shape[2]

        assert len(image_to_overwrite) == past_length

        logits = output.logits[0, -1]
        del output
        input_ids = logits.argmax().view(1, 1)
        attention_mask = torch.ones((1, past_length+1), device=self.llava.device, dtype=torch.bool)
        output = self.llava.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            **kwargs)

        output_ids = output.sequences[0, :-1]   # the last token was not passed through the model
        assert input_ids[0] == logits.argmax()
        attentions = output.attentions   # output_len, num_layers, (1/bs, num_heads, 1/seq_len, cur_len)
        assert len(attentions) == len(output_ids)
        assert len(attentions[0]) == self.llava.config.text_config.num_hidden_layers
        num_layers = len(attentions[0])
        attentions = [torch.cat([attn[layer_id][0, ..., :past_length][..., image_to_overwrite]
                                 for attn in attentions], dim=-2) for layer_id in range(num_layers)]
        hidden_states = output.hidden_states   # output_len, num_layers + 1, bs/1, seq_len/1, hidden_dim
        hidden_states = [torch.cat([feat[layer_id] for feat in hidden_states], dim=-2)
                         for layer_id in range(1, 1+num_layers)]
        # do keyword detection
        text_layer_weights = self.get_text_layer_weights()
        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        key_phrases = self.key_phrase_head(hidden_states)  # num_key_phrases, answer_len
        import pdb; pdb.set_trace()
        if len(key_phrases) == 0:
            key_phrases = torch.ones((1, len(output_ids)),
                                     device=self.llava.device, dtype=torch.bool)
        mask_attentions = []
        key_phrase_ids = []
        text_embeds = []
        for key_phrase in key_phrases:
            if key_phrase.sum() == 0:
                key_phrase = torch.ones_like(key_phrase)

            matched_len = key_phrase.sum()
            if self.merge == 'learn':
                merge_weights = self.merge_proj(hidden_states[key_phrase])[:, 0].softmax(0)   # matched_len
                merge_weights = merge_weights.view(1, matched_len, 1)
            else:
                merge_weights = None

            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, key_phrase], dim=1, weights=merge_weights) for attn in attentions]))
            key_phrase_ids.append(output_ids[key_phrase])
            text_embeds.append(self.text_proj(hidden_states[key_phrase]))
        del attentions
        # import pdb; pdb.set_trace()
        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)
        meta_data = data_sample['meta_data']
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size
        mask_attentions = mask_attentions.view(*mask_attentions.shape[:-1], llava_h, llava_w)
        pred_masks = self.mask_head(mask_attentions)[:, 0]
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)
        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)

        pred_masks = pred_masks[:, before_height:before_height + mask_h,
                                   before_width:before_width + mask_w].contiguous()
        pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)

        height, width = data_sample['height'], data_sample['width']
        pred_masks = F.interpolate(pred_masks[None], size=(height, width), mode='bilinear')[0].cpu()
        pred_masks = pred_masks > 0

        return output_ids, key_phrase_ids, pred_masks

    @torch.no_grad()
    def caption_forward(self, data_sample, **kwargs):
        # for now we implement greedy search only
        input_ids = data_sample['input_ids'][None].to(self.llava.device)
        pixel_values = data_sample['pixel_values'][None].to(device=self.llava.device,
                                                            dtype=self.llava.dtype)
        attention_mask = torch.ones(input_ids.shape, device=self.llava.device, dtype=torch.bool)
        output0 = self.llava.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            # output_attentions=True,
            # output_hidden_states=True,
            # return_dict_in_generate=True,
            **kwargs)[0]
        output1 = self.llava.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs).sequences[0]
        # import pdb; pdb.set_trace()

        output = self.llava(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            use_cache=True)
        past_key_values = output.past_key_values
        past_length = past_key_values[0][0].shape[2]
        logits = output.logits[0, -1]
        del output
        input_ids = logits.argmax().view(1, 1)
        attention_mask = torch.ones((1, past_length+1), device=self.llava.device, dtype=torch.bool)
        output2 = self.llava.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs)[0]

        output3 = self.llava.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            **kwargs).sequences[0]

        return output0, output1, output2, output3
