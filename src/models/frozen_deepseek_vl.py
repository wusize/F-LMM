import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import LoadWoInit
from mmengine.logging import print_log


@torch.no_grad()
def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection / (union + 1e-12)


class FrozenDeepseekVL(BaseModel):

    def __init__(self,
                 model,
                 tokenizer,
                 mask_head,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 **kwargs):
        super().__init__()
        with LoadWoInit():
            self.deepseek_vl = BUILDER.build(model)
        self.deepseek_vl.requires_grad_(False)
        self.tokenizer = BUILDER.build(tokenizer)
        self.image_token_idx = self.tokenizer.encode('<image_placeholder>', add_special_tokens=False)[-1]
        print_log(f"Image token: {self.tokenizer.decode(self.image_token_idx)}")
        in_channels = (self.deepseek_vl.config.language_config.num_attention_heads *
                       self.deepseek_vl.config.language_config.num_hidden_layers)
        mask_head.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.merge = merge
        assert merge in ['mean', 'max']
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        self.patch_size = 16   # hard-code use siglip_large_patch16_384
        self.clip_shape = 24
        self._generation_ready = False

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
        self.deepseek_vl.train(mode=False)
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


class FrozenDeepseekVLSAM(FrozenDeepseekVL):
    def __init__(self, sam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        self.text_proj = nn.Linear(self.deepseek_vl.config.language_config.hidden_size,
                                   self.sam.model.prompt_encoder.embed_dim)
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.deepseek_vl.config.language_config.num_hidden_layers))

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def _forward(self, data_sample):
        text_layer_weights = self.get_text_layer_weights()
        pixel_values = data_sample['pixel_values'][None, None].to(
            device=self.deepseek_vl.device,
            dtype=self.deepseek_vl.dtype)
        input_ids = data_sample['input_ids'][None].to(self.deepseek_vl.device)
        images_seq_mask = input_ids == self.image_token_idx
        images_emb_mask = torch.ones((1, 1, images_seq_mask.sum()), dtype=torch.bool,
                                     device=self.deepseek_vl.device)

        inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)

        with torch.no_grad():
            outputs = self.deepseek_vl.language_model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                use_cache=False)

        mask_ids = data_sample['mask_ids'].to(self.deepseek_vl.device)
        meta_data = data_sample['meta_data']
        attentions = [attn[0, ..., images_seq_mask[0]] for attn in outputs.attentions]
        attentions = [attn.view(*attn.shape[:-1], self.clip_shape, self.clip_shape) for attn in attentions]
        hidden_states = outputs.hidden_states[-self.deepseek_vl.config.language_config.num_hidden_layers:]
        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        del outputs

        masks = data_sample['masks']
        mask_attentions = []
        text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            mask_attentions.append(
                torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]
                )
            )
            text_embeds.append(self.text_proj(hidden_states[matched]))
        del attentions
        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)

        pred_masks = self.mask_head(mask_attentions)[:, 0]
        # todo: unpad pred_masks
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)

        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()
        sam_pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)

        output = dict(pred_masks=pred_masks, sam_pred_masks=sam_pred_masks,
                      mask_ids=mask_ids, hidden_states=hidden_states)

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
            masks = data_sample['masks'].to(self.deepseek_vl.device)
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
                                prompt_template,
                                max_thought_tokens,
                                max_new_tokens,
                                **kwargs):
        from transformers import StoppingCriteriaList
        from xtuner.utils import StopWordStoppingCriteria
        if isinstance(image_processor, dict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        self.prompt_template = prompt_template
        self.max_thought_tokens = max_thought_tokens
        self.max_new_tokens = max_new_tokens

        stop_words = self.prompt_template.get('STOP_WORDS', []) + ['.']   # only need the first sentence
        self.stop_criteria = StoppingCriteriaList()
        self.stop_word_ids = [self.tokenizer.encode(word, add_special_tokens=False)[-1]
                              for word in stop_words]
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))
        self._generation_ready = True

    @torch.no_grad()
    def visual_cot_v1(self, image, question):
        # v1: let the llm first describe the most relevant object
        assert self._generation_ready
        # 1. Round one: prompt the llm to find the most relevant object
        prompt = self.prompt_template['INSTRUCTION'].format(
            input='<image_placeholder>' + question + 'First think which object in this image is most relevant to the question.')
        prompt += ' The object most relevant to the question is'
        assert prompt.count('<image_placeholder>') == 1
        prompt = prompt.replace('<image_placeholder>', '<image_placeholder>' * 576)

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.deepseek_vl.device)
        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data['pixel_values'][0]
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values[None, None].to(
            device=self.deepseek_vl.device, dtype=self.deepseek_vl.dtype)
        images_seq_mask = input_ids == self.image_token_idx
        assert images_seq_mask.sum() == 576
        images_emb_mask = torch.ones((1, 1, 576), dtype=torch.bool,
                                     device=self.deepseek_vl.device)
        inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)
        outputs = self.deepseek_vl.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_thought_tokens,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            output_attentions=True,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
        output_ids = outputs.sequences[0, :-1]    # discard the last one
        thought = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        past_key_values = outputs.past_key_values
        assert len(output_ids) + inputs_embeds.shape[1] == past_key_values[0][0].shape[2]
        attentions = outputs.attentions[1:]
        hidden_states = outputs.hidden_states[1:]
        assert len(output_ids) == len(attentions)
        assert len(output_ids) == len(hidden_states)

        # 2. locate the object
        images_seq_indices = torch.where(images_seq_mask[0])[0]
        attentions = [torch.cat([attn[layer_id][0, ..., images_seq_indices] for attn in attentions], dim=-2)
                      for layer_id in range(self.deepseek_vl.language_model.config.num_hidden_layers)]
        attentions = [attn.view(*attn.shape[:-1], self.clip_shape, self.clip_shape) for attn in attentions]
        # num_layers, (num_heads, seq_len, h, w)
        text_layer_weights = self.get_text_layer_weights()
        hidden_states = torch.stack(
            [torch.cat([hs[layer_id+1][0] for hs in hidden_states], dim=-2)
             for layer_id in range(self.deepseek_vl.language_model.config.num_hidden_layers)
             ])   # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        mask_attentions = torch.cat([self.apply_merge(attn, dim=1)
                                     for attn in attentions])[None].to(self.mask_head.dtype)
        text_embeds = self.text_proj(hidden_states)[None]

        pred_masks = self.mask_head(mask_attentions)[:, 0]
        # todo: unpad pred_masks
        meta_data = image_data['meta_datas'][0]
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)

        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()
        pred_mask = self.sam(image, pred_masks, text_embeds)[0]
        bbox = self.mask2box(pred_mask > 0.0)

        # 3. crop the object from the image and answer the question
        image = image.crop(bbox)
        prompt = self.prompt_template['INSTRUCTION'].format(
            input='<image_placeholder>' * 576 + 'This is the cropped image region of the object. Now answer the '
                                                'question using a single word or phrase.')
        prompt = self.tokenizer.eos_token + prompt   # eos is to end the previous answer
        input_ids = self.tokenizer.encode(prompt,
                                          add_special_tokens=False,   # bos not needed
                                          return_tensors='pt').to(self.deepseek_vl.device)
        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data['pixel_values'][0]
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values[None, None].to(
            device=self.deepseek_vl.device, dtype=self.deepseek_vl.dtype)
        images_seq_mask = input_ids == self.image_token_idx
        assert images_seq_mask.sum() == 576
        images_emb_mask = torch.ones((1, 1, 576), dtype=torch.bool,
                                     device=self.deepseek_vl.device)
        inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)
        ## 3.1 pass the inputs first as the inputs_embeds conflict with past_key_values in generation
        outputs = self.deepseek_vl.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True)

        output_ids = outputs.logits[:, -1:].argmax(dim=-1)    # the first id of the answer
        ## 3.2 generate final answer
        all_output_ids = [output_ids[0, 0].item()]
        while len(all_output_ids) < self.max_new_tokens:
            outputs = self.deepseek_vl.language_model(
                input_ids=output_ids,
                past_key_values=outputs.past_key_values,
                return_dict=True,
                use_cache=True)
            output_ids = outputs.logits.argmax(dim=-1)  # the first id of the answer
            assert output_ids.shape[1] == 1
            if output_ids[0, 0].item() in self.stop_word_ids:
                break
            all_output_ids.append(output_ids[0, 0].item())

        # TODO: fix the bug in using generate function
        # output_ids = self.deepseek_vl.language_model.generate(
        #     input_ids=start_id,
        #     attention_mask=torch.ones(1, 1 + past_key_values[0][0].shape[2],
        #                               device=self.deepseek_vl.device,
        #                               dtype=torch.long),
        #     past_key_values=past_key_values,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     max_new_tokens=self.max_new_tokens,
        #     do_sample=False,
        #     use_cache=True,
        # )[0]

        answer = self.tokenizer.decode(all_output_ids, skip_special_tokens=True)

        return thought, bbox, answer

    @torch.no_grad()
    def visual_cot_v2(self, image, question):
        # v2: dirrecly ground the whole question
        assert self._generation_ready
        prompt = self.prompt_template['INSTRUCTION'].format(
            input='<image_placeholder>' * 576 + question + '<image_placeholder>')  # temporarily insert this special token to locate the question
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.deepseek_vl.device)
        image_places = torch.where(input_ids[0] == self.image_token_idx)[0]
        question_start_place = image_places[-2] + 1
        question_end_place = image_places[-1]

        # 1. locate the question
        cur_input_ids = input_ids[:, :question_end_place]
        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data['pixel_values'][0]
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values[None, None].to(
            device=self.deepseek_vl.device, dtype=self.deepseek_vl.dtype)
        images_seq_mask = cur_input_ids == self.image_token_idx
        assert images_seq_mask.sum() == 576
        images_emb_mask = torch.ones((1, 1, 576), dtype=torch.bool,
                                     device=self.deepseek_vl.device)
        inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=cur_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)
        outputs = self.deepseek_vl.language_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            use_cache=True)
        past_key_values = outputs.past_key_values

        text_layer_weights = self.get_text_layer_weights()
        attentions = [attn[0, ..., images_seq_mask[0]] for attn in outputs.attentions]
        attentions = [attn.view(*attn.shape[:-1], self.clip_shape, self.clip_shape) for attn in attentions]
        hidden_states = outputs.hidden_states[-self.deepseek_vl.config.language_config.num_hidden_layers:]
        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        text_embeds = self.text_proj(hidden_states[question_start_place:])[None]
        mask_attentions = torch.cat(
            [self.apply_merge(attn[:, question_start_place:], dim=1) for attn in attentions]
        )[None].to(self.mask_head.dtype)

        pred_masks = self.mask_head(mask_attentions)[:, 0]
        # todo: unpad pred_masks
        meta_data = image_data['meta_datas'][0]
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)

        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()
        pred_mask = self.sam(image, pred_masks, text_embeds)[0]

        # 2. append the cropped image
        bbox = self.mask2box(pred_mask > 0.0)
        image = image.crop(bbox)

        inserted_input_ids = self.tokenizer.encode(
            '<image_placeholder>' * 576 + 'This is the cropped image region of the object relevant '
                                          'to the above question. Now answer the question using a '
                                          'single word or phrase.',
            return_tensors='pt', add_special_tokens=False).to(self.deepseek_vl.device)

        appended_input_ids = torch.cat(
            [inserted_input_ids, input_ids[:, question_end_place + 1:]], dim=1)

        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data['pixel_values'][0]
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values[None, None].to(
            device=self.deepseek_vl.device, dtype=self.deepseek_vl.dtype)
        images_seq_mask = appended_input_ids == self.image_token_idx
        assert images_seq_mask.sum() == 576
        images_emb_mask = torch.ones((1, 1, 576), dtype=torch.bool,
                                     device=self.deepseek_vl.device)
        appended_inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=appended_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)

        ## 3.1 pass the inputs first as the inputs_embeds conflict with past_key_values in generation
        outputs = self.deepseek_vl.language_model(
            inputs_embeds=appended_inputs_embeds,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True)

        output_ids = outputs.logits[:, -1:].argmax(dim=-1)    # the first id of the answer
        ## 3.2 generate final answer
        all_output_ids = [output_ids[0, 0].item()]
        while len(all_output_ids) < self.max_new_tokens:
            outputs = self.deepseek_vl.language_model(
                input_ids=output_ids,
                past_key_values=outputs.past_key_values,
                return_dict=True,
                use_cache=True)
            output_ids = outputs.logits.argmax(dim=-1)  # the first id of the answer
            assert output_ids.shape[1] == 1
            if output_ids[0, 0].item() in self.stop_word_ids:
                break
            all_output_ids.append(output_ids[0, 0].item())

        answer = self.tokenizer.decode(all_output_ids, skip_special_tokens=True)

        return '', bbox, answer

    @staticmethod
    def mask2box(mask, scale=1.0):
        h, w = mask.shape
        assert mask.dtype == torch.bool
        ys, xs = torch.where(mask)
        if len(ys) == 0:
            return 0, 0, w, h
        else:
            y0, y1 = ys.min().item(), ys.max().item()
            x0, x1 = xs.min().item(), xs.max().item()

            yd, xd = max((y1 - y0) / 2, 8), max((x1 - x0) / 2, 8)
            yc, xc = (y1 + y0) / 2, (x1 + x0) / 2

            x0, x1 = max(0, xc - xd * scale), min(w, xc + xd * scale)
            y0, y1 = max(0, yc - yd * scale), min(h, yc + yd * scale)

            return int(x0), int(y0), int(x1), int(y1)

    @torch.no_grad()
    def visual_cot_v3(self, image, question):
        # v3: the baseline, no cot
        assert self._generation_ready
        prompt = self.prompt_template['INSTRUCTION'].format(
            input='<image_placeholder>' * 576 + question +
                  ' Answer the question using a single word or phrase.')
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.deepseek_vl.device)
        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data['pixel_values'][0]
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values[None, None].to(
            device=self.deepseek_vl.device, dtype=self.deepseek_vl.dtype)
        images_seq_mask = input_ids == self.image_token_idx
        assert images_seq_mask.sum() == 576
        images_emb_mask = torch.ones((1, 1, 576), dtype=torch.bool,
                                     device=self.deepseek_vl.device)
        inputs_embeds = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)

        output_ids = self.deepseek_vl.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=self.stop_criteria,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )[0]

        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return '', (0, 0, image.width, image.height), answer


if __name__ == '__main__':
    from PIL import Image
    from xtuner.model.utils import guess_load_checkpoint
    from mmengine.config import Config
    image = Image.open('images/dog_a.png')
    question = "<image_placeholder>What category does the dog belong to?"
    cfg = Config.fromfile('configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py')
    model = BUILDER.build(cfg.model)
    _ = model.load_state_dict(guess_load_checkpoint('checkpoints/frozen_deepseek_vl_1_3b_unet_sam_l_iter_95080.pth'),
                              strict=False)
    model._prepare_for_generation(image_processor=cfg.image_processor,
                                  prompt_template=cfg.prompt_template,
                                  max_thought_tokens=16,
                                  max_new_tokens=512)
    model = model.cuda().eval()
    output = model.visual_cot_v1(image, question)
