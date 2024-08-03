import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import guess_load_checkpoint
from flmm.utils import compute_mask_IoU
from xtuner.utils import DEFAULT_IMAGE_TOKEN, StopWordStoppingCriteria
from transformers import StoppingCriteriaList


class FrozenLlava(BaseModel):

    def __init__(self,
                 model,
                 mask_head,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 pretrained=None,
                 **kwargs):
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
        assert merge in ['mean', 'max']

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llava.config.text_config.num_hidden_layers))

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)
        
    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

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
        super().train(mode=mode)
        self.llava.train(mode=False)
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
                      for attn in outputs.attentions]   # num_layers, (1, num_heads, seq_len, seq_len)
        # -> num_layers, (num_heads, seq_len, h*w)
        hidden_states = outputs.hidden_states[-self.llava.config.text_config.num_hidden_layers:]
        # num_layers (1, seq_len, dim)

        labels = outputs.labels[0]

        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        del outputs

        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size

        attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
        # -> num_layers, (num_heads, seq_len, h, w)
        masks = data_sample['masks']
        mask_attentions = []
        text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            mask_attentions.append(torch.cat(    # num_layers * num_heads, * , h, w)
                [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]))
            text_embeds.append(self.text_proj(hidden_states[matched]))

        del attentions
        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)
        # if self.training:
        #     mask_attentions.requires_grad = True
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

        return loss_dict

    def _prepare_for_generation(self,
                                image_processor,
                                tokenizer,
                                prompt_template,
                                max_new_tokens=16,
                                **kwargs):

        if isinstance(tokenizer, dict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        if isinstance(image_processor, dict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens

        stop_words = self.prompt_template.get('STOP_WORDS', [])
        self.stop_criteria = StoppingCriteriaList()
        self.stop_word_ids = [self.tokenizer.encode(word, add_special_tokens=False)[-1]
                              for word in stop_words]
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))
        self._generation_ready = True
        self.config = self.llava.config

    def reason_seg(self, image, instruction, answer_prefix=None):
        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data['pixel_values'][0]
        meta_data = image_data['meta_datas'][0]
        prompt = self.prompt_template['INSTRUCTION'].format(
            input=DEFAULT_IMAGE_TOKEN + '\n' + instruction)
        if answer_prefix is not None:
            prompt += answer_prefix

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.llava.device)
        pixel_values = torch.from_numpy(pixel_values)[None].to(device=self.llava.device,
                                                               dtype=self.llava.dtype)

        with torch.inference_mode():
            outputs = self.llava(**inputs,
                                 pixel_values=pixel_values,
                                 use_cache=True,
                                 return_dict=True)

        past_key_values = outputs.past_key_values
        image_to_overwrite = outputs.image_to_overwrite
        input_ids = outputs.logits[0, -1].argmax().view(1, 1)
        attention_mask = torch.ones(1, image_to_overwrite.shape[1]+1, device=self.llava.device,
                                    dtype=torch.bool)
        input_ids = torch.cat([
            torch.ones(1, image_to_overwrite.shape[1], device=self.llava.device, dtype=torch.long),
            input_ids
        ], dim=1)

        with torch.inference_mode():
            outputs = self.llava.language_model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.eos_token_id,
                stopping_criteria=self.stop_criteria,
                output_attentions=True,
                output_hidden_states=True,
            )

        num_hidden_layers = self.llava.config.text_config.num_hidden_layers
        image_places = torch.where(image_to_overwrite[0])[0]
        attentions = outputs.attentions   # cur_seq_len, num_layers, (1, num_heads, 1, past_seq_len)

        attentions = [torch.cat([attn[layer_id][0, ..., image_places] for attn in attentions], dim=-2)
                      for layer_id in range(num_hidden_layers)]
        # num_layers, (num_heads, cur_seq_len, h*w)
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size
        mask_attentions = self.apply_merge(torch.cat(attentions), dim=1).view(1, -1, llava_h, llava_w)

        hidden_states = outputs.hidden_states  # cur_seq_len, num_layers+1, (1, 1, dim)
        text_layer_weights = self.get_text_layer_weights()

        hidden_states = [torch.cat([h[layer_id][0] for h in hidden_states])  # cur_seq_len, dim
                         for layer_id in range(1, 1+num_hidden_layers)]
        hidden_states = torch.stack(hidden_states)
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        pred_masks = self.mask_head(mask_attentions.to(self.mask_head.dtype))[:, 0]
        # todo: unpad pred_masks
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]

        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)

        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()
        sam_pred_masks = self.sam(image, pred_masks, [self.text_proj(hidden_states)])
        answer = self.tokenizer.decode(outputs.sequences[0, image_to_overwrite.shape[1]:],
                                       skip_special_tokens=True)


        return answer, sam_pred_masks[0] > 0
