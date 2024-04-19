import numpy as np
import torch
import torch.nn as nn
from transformers import GenerationConfig, StoppingCriteriaList
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria
from pycocotools import mask as mask_utils



def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle


def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out


class GCGWrapper(nn.Module):
    def __init__(self, config_file, max_new_tokens=100):
        super(GCGWrapper, self).__init__()
        self.config = Config.fromfile(config_file)
        self.model = BUILDER.build(self.config.model)
        self.tokenizer = BUILDER.build(self.config.tokenizer)
        self.model.eval()

        stop_words = self.config.prompt_template.get('STOP_WORDS', [])
        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )

    def load_pretrained(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def forward(self, data_sample):
        output_ids, key_phrase_ids, pred_masks = self.model.gcg_forward(
            data_sample,
            generation_config=self.gen_config,
            stopping_criteria=self.stop_criteria
        )
        output_ids_debug = self.model.caption_forward(data_sample,
                                                      generation_config=self.gen_config,
                                                      stopping_criteria=self.stop_criteria
                                                      )
        caption = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        caption_debug = self.tokenizer.decode(output_ids_debug, skip_special_tokens=True)
        import pdb; pdb.set_trace()

        phrases = [self.tokenizer.decode(key_phrase_ids_, skip_special_tokens=True)
                   for key_phrase_ids_ in key_phrase_ids]

        uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks)
        rle_masks = []
        for m in uncompressed_mask_rles:
            rle_masks.append(coco_encode_rle(m))

        # Create results dictionary
        result_dict = {
            "image_id": data_sample['image_id'],
            "caption": caption,
            "phrases": phrases,
            "pred_masks": rle_masks
        }

        return result_dict
