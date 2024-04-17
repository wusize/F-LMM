import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from xtuner.registry import BUILDER
from mmdet.models import DetrTransformerDecoder


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class KeyPhraseHead(nn.Module):
    def __init__(self,
                 max_num=50,
                 temp=10.0,
                 in_channels=256,
                 decoder=None,
                 loss_mask=None,
                 loss_dice=None):
        super(KeyPhraseHead, self).__init__()
        self.decoder = DetrTransformerDecoder(**decoder)
        embed_dim = self.decoder.embed_dims
        self.key_phrase_queries = nn.Parameter(torch.randn(max_num, embed_dim))
        self.max_num = max_num
        self.embed_dim = embed_dim
        self.temp = nn.Parameter(torch.tensor(temp))
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_temperature(self):
        return self.temp.clamp(min=1.0)

    def forward(self, hidden_states, labels=None):
        hidden_states = self.in_proj(hidden_states.detach())    # detach to make it an independent branch
        seq_len, _ = hidden_states.shape

        query = torch.zeros_like(self.key_phrase_queries)
        key = hidden_states
        value = hidden_states
        query_pos = self.key_phrase_queries
        key_pos = positional_encoding_1d(self.embed_dim, seq_len).to(device=self.device,
                                                                     dtype=self.dtype)



        key_phrase_queries = F.normalize(q[seq_len:], dim=-1)
        text_seq_queries = F.normalize(q[:seq_len], dim=-1)
        logits = self.get_temperature() * key_phrase_queries @ text_seq_queries
        if labels is None:
            pred_masks = logits > 0.0
            pred_masks = pred_masks[pred_masks.sum(dim=-1) > 0]
            return pred_masks
        else:
            label_ids = torch.unique(labels)
            gt_masks = torch.stack([labels == label_id
                                    for label_id in label_ids if label_id >= 0]).to(self.dtype)
            assert len(gt_masks) > 0
            return self.loss(logits, gt_masks)

    @property
    def dtype(self):
        return self.temp.dtype

    @property
    def device(self):
        return self.temp.device

    def loss(self, logits, gt_masks):
        with torch.no_grad():
            pred_masks = (logits > 0.0).to(self.dtype)
            mask_prod = gt_masks[:, None] * pred_masks[None]   # num_gt, num_pred, seq_len
            mask_sum = gt_masks[:, None] + pred_masks[None]
            intersection = mask_prod.sum(dim=-1)   # num_gt, num_pred
            union = (mask_sum - mask_prod).sum(dim=-1)   # num_gt, num_pred
            ious = (intersection / (union + 1e-12)).float().detach().cpu().numpy()
            row_ids, col_ids = linear_sum_assignment(-ious)
            # todo: lift this restriction
            assert len(row_ids) == len(gt_masks), f"row_ids:{row_ids.shape}, gt_masks:{gt_masks.shape}"
            aiou = torch.from_numpy(ious[row_ids, col_ids]).to(device=self.device,
                                                               dtype=self.dtype).mean()
            del pred_masks

        pos_gt_masks = gt_masks[row_ids.tolist()]
        pos_logits = logits[col_ids.tolist()]
        loss_dice = self.loss_dice(pos_logits, pos_gt_masks, avg_factor=pos_logits.shape[0])
        loss_mask_pos = self.loss_mask(pos_logits.view(-1), pos_gt_masks.view(-1),
                                       avg_factor=pos_logits.numel())

        pos_pred = torch.zeros(len(logits), device=self.device, dtype=torch.bool)
        pos_pred[col_ids.tolist()] = True
        neg_logits = logits[torch.logical_not(pos_pred)]
        neg_gt_masks = torch.zeros(len(neg_logits), device=self.device, dtype=self.dtype)
        if len(neg_logits) > 0:
            loss_mask_neg = self.loss_mask(neg_logits.view(-1), neg_gt_masks.view(-1),
                                           avg_factor=neg_logits.numel())
        else:
            loss_mask_neg = loss_mask_pos

        loss_mask = (loss_mask_pos + loss_mask_neg) * 0.5

        return loss_dice, loss_mask, aiou
