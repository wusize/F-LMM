import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import LoadWoInit
from mmengine.logging import print_log
# from xtuner.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from flmm.utils import compute_mask_IoU


class FrozenMGM(BaseModel):

    def __init__(self,
                 model,
                 mask_head,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 **kwargs):
        super().__init__()
        self._init_mgm_model(model)
        in_channels = self.mgm.config.num_attention_heads * self.mgm.config.num_hidden_layers
        if (getattr(self.mgm.config, 'image_grid', 1) > 1
                and getattr(self.mgm.config, 'image_global', False)):
            in_channels *= 2
        mask_head.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
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
        super().train(mode=mode)
        self.mgm.train(mode=False)
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

    @property
    def patch_size(self):
        return self.mgm.get_vision_tower().config.patch_size

    @property
    def clip_shape(self):
        return (self.image_processor.image_size_raw['height'] // self.patch_size,
                self.image_processor.image_size_raw['width'] // self.patch_size)

    def _init_mgm_model(self, model):
        model_path = model['pretrained_model_name_or_path']
        with LoadWoInit():
            model = BUILDER.build(model)
        print_log("Finished loading MGM model")
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            print_log("Finished loading CLIP vision model")
        vision_tower.to(dtype=model.dtype)
        image_processor = vision_tower.image_processor

        vision_tower_aux = model.get_vision_tower_aux()
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()
            print_log("Finished loading Convnext model")
        vision_tower_aux.to(dtype=model.dtype)

        # initialize attention modules
        model.config.model_path = model_path
        model.get_model().initialize_uni_modules(model.config, for_eval=True)

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy()
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux

        self.context_len = context_len
        self.mgm = model
        self.image_processor = image_processor
        self.mgm.requires_grad_(False)

    def _process_image(self, image):

        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']

        image_grid = getattr(self.mgm.config, 'image_grid', 1)
        if hasattr(self.mgm.config, 'image_size_aux'):
            raw_shape = [self.image_processor.image_size_raw['height'] * image_grid,
                         self.image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                           size=raw_shape,
                                                           mode='bilinear',
                                                           align_corners=False)
        else:
            image_tensor_aux = []

        if image_grid >= 2:
            raw_image = image_tensor.reshape(3,
                                             image_grid,
                                             self.image_processor.image_size_raw['height'],
                                             image_grid,
                                             self.image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3,
                                          self.image_processor.image_size_raw['height'],
                                          self.image_processor.image_size_raw['width'])

            if getattr(self.mgm.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image,
                                                               size=[self.image_processor.image_size_raw['height'],
                                                                     self.image_processor.image_size_raw['width']],
                                                               mode='bilinear',
                                                               align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.mgm.device, dtype=self.mgm.dtype)
        image_tensor_aux = image_tensor_aux.to(self.mgm.device, dtype=self.mgm.dtype)

        return image_tensor, image_tensor_aux

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

    def _process_attention(self, attention2image):
        num_heads, seq_len, image_len = attention2image.shape
        single_image_len = self.clip_shape[0] * self.clip_shape[1]
        image_grid = getattr(self.mgm.config, 'image_grid', 1)
        if image_grid == 1:
            assert image_len == single_image_len
            return attention2image.view(num_heads, seq_len, *self.clip_shape)
        else:
            use_global_image = getattr(self.mgm.config, 'image_global', False)
            if use_global_image:
                assert image_len == (1 + image_grid ** 2) * single_image_len
                attention2global_image = attention2image[..., :single_image_len]
                attention2global_image = attention2global_image.view(num_heads, seq_len, *self.clip_shape)
                attention2hd_image = attention2image[..., single_image_len:]
            else:
                assert image_len == (image_grid ** 2) * single_image_len
                attention2global_image = None
                attention2hd_image = attention2image

            attention2hd_image = attention2hd_image.view(num_heads, seq_len, image_grid, image_grid,
                                                         *self.clip_shape)
            attention2hd_image = attention2hd_image.permute(0, 1, 2, 4, 3, 5).contiguous()
            attention2hd_image = attention2hd_image.view(num_heads, seq_len, image_grid * self.clip_shape[0],
                                                         image_grid * self.clip_shape[1])

            if attention2global_image is not None:
                attention2global_image = F.interpolate(attention2global_image.float(),
                                                       scale_factor=image_grid,
                                                       mode='bilinear').to(attention2global_image)
                attention2hd_image = torch.cat([attention2global_image, attention2hd_image], dim=0)

            return attention2hd_image


class FrozenMGMSAM(FrozenMGM):
    def __init__(self, sam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        self.text_proj = nn.Linear(self.mgm.config.hidden_size,
                                   self.sam.model.prompt_encoder.embed_dim)
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.mgm.config.num_hidden_layers))

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def _forward(self, data_sample):
        text_layer_weights = self.get_text_layer_weights()

        image_tensor, image_tensor_aux = self._process_image(data_sample['pixel_values'])
        input_ids = data_sample['input_ids'][None].to(self.mgm.device)
        mask_ids = data_sample['mask_ids'][None].to(self.mgm.device)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            outputs = self.mgm(input_ids=input_ids,
                               mask_ids=mask_ids,
                               images=image_tensor,
                               images_aux=image_tensor_aux,
                               output_hidden_states=True,
                               output_attentions=True,
                               return_dict=True,
                               use_cache=False)

        # import pdb; pdb.set_trace()

        meta_data = data_sample['meta_data']
        mask_ids = outputs.mask_ids[0]
        attentions = [attn[0, ..., outputs.image_places[0]]
                      for attn in outputs.attentions]
        hidden_states = outputs.hidden_states[-self.mgm.config.num_hidden_layers:]
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
                    [self.apply_merge(
                        self._process_attention(attn[:, matched]),
                        dim=1) for attn in attentions]
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
            masks = data_sample['masks'].to(self.mgm.device)
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
                                tokenizer,
                                prompt_template,
                                max_new_tokens=512,
                                **kwargs):
        raise NotImplementedError
