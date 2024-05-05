import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import LoadWoInit
from mmengine.logging import print_log
from xtuner.model.llava import prepare_inputs_labels_for_multimodal
from xtuner.utils.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX


@torch.no_grad()
def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection / (union + 1e-12)


class FrozenHPT(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 projector,
                 mask_head,
                 visual_select_layer=-2,
                 image_size=588,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 **kwargs):
        super().__init__()
        self.visual_select_layer = visual_select_layer
        self.image_size = image_size
        self._init_models(llm=llm, visual_encoder=visual_encoder, projector=projector)
        in_channels = (self.llm.config.num_attention_heads *
                       self.llm.config.num_hidden_layers)
        mask_head.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.merge = merge
        assert merge in ['mean', 'max']
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

    @staticmethod
    def interpolate_pos_embed(model, new_size):
        import pdb; pdb.set_trace()
        pos_emb = model.vision_model.embeddings.position_embedding.weight.float()
        ori_size = int((pos_emb.shape[0] - 1) ** 0.5)
        dim = pos_emb.shape[1]
        print_log("Position interpolate from %dx%d to %dx%d" % (ori_size, ori_size, new_size, new_size))
        extra_tokens = pos_emb[0, :][None]
        pos_tokens = pos_emb[1:, :]
        pos_tokens = pos_tokens.reshape(-1, ori_size, ori_size, dim).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        new_pos_embed = new_pos_embed.to(torch.float16)
        return torch.nn.Parameter(new_pos_embed)

    def _init_models(self, llm, visual_encoder, projector):
        with LoadWoInit():
            llm = BUILDER.build(llm)
            visual_encoder = BUILDER.build(visual_encoder)
            projector = BUILDER.build(projector)
        import pdb; pdb.set_trace()
        patch_size = visual_encoder.vision_model.embeddings.patch_size
        num_positions = (self.image_size // patch_size) ** 2 + 1
        new_size = self.image_size // patch_size
        visual_encoder.vision_model.embeddings.num_patches = (self.image_size // patch_size) ** 2
        visual_encoder.vision_model.embeddings.num_positions = num_positions
        visual_encoder.vision_model.embeddings.position_ids = torch.arange(num_positions).expand((1, -1))
        visual_encoder.vision_model.embeddings.position_embedding.weight = self.interpolate_pos_embed(
            visual_encoder, new_size)
        visual_encoder.config.image_size = self.image_size
        self.clip_shape = new_size
        self.llm = llm
        self.visual_encoder = visual_encoder.to(llm.dtype)
        self.projector = projector.to(llm.dtype)
        self.llm.requires_grad_(False)
        self.visual_encoder.requires_grad_(False)
        self.projector.requires_grad_(False)

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
        self.llm.train(mode=False)
        self.visual_encoder.train(mode=False)
        self.projector.train(mode=False)
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


class FrozenHPTSAM(FrozenHPT):
    def __init__(self, sam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        self.text_proj = nn.Linear(self.llm.config.hidden_size,
                                   self.sam.model.prompt_encoder.embed_dim)
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llm.config.num_hidden_layers))

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def _forward(self, data_sample):
        text_layer_weights = self.get_text_layer_weights()
        with torch.no_grad():
            pixel_values = data_sample['pixel_values'][None].to(device=self.visual_encoder.device,
                                                                dtype=self.visual_encoder.dtype)
            visual_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:].to(self.projector.dtype)
            ).to(self.llm.dtype)

        input_ids = data_sample['input_ids'][None].to(self.llm.device)
        mask_ids = data_sample['mask_ids'][None].to(self.llm.device)

        # treat mask_ids as labels todo: simplify previously implementations by doing so in other models (if time permits)
        mask_ids[input_ids == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=input_ids, pixel_values=pixel_values, labels=mask_ids)
        mask_ids = mm_inputs.pop('labels')
        image_places = mask_ids[0] == IGNORE_INDEX

        with torch.no_grad():
            outputs = self.llm(
                **mm_inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                use_cache=False)

        attentions = [attn[0, ..., image_places] for attn in outputs.attentions]
        attentions = [attn.view(*attn.shape[:-1], self.clip_shape, self.clip_shape) for attn in attentions]
        hidden_states = outputs.hidden_states[-self.llm.config.num_hidden_layers:]
        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        del outputs

        masks = data_sample['masks']
        mask_attentions = []
        text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids[0] == mask_id
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
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        meta_data = data_sample['meta_data']
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
            masks = data_sample['masks'].to(self.llm.device)
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

