import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.utils.constants import IGNORE_INDEX
from xtuner.model.utils import LoadWoInit

@torch.no_grad()
def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection / (union + 1e-12)


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
        mask_head.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mask_head)
        self.merge = merge
        assert merge in ['mean', 'max']
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.text_layer_weights = nn.Parameter(
            torch.ones(self.fuyu.config.num_hidden_layers))

    def _init_mgm_model(self, model):
        model_path = model['pretrained_model_name_or_path']
        with LoadWoInit():
            model = BUILDER.build(model)
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=model.dtype)
        image_processor = vision_tower.image_processor

        vision_tower_aux = model.get_vision_tower_aux()
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()
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

        image_tensor = self.image_processor.preprocess(image.numpy(), return_tensors='pt')['pixel_values']

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
