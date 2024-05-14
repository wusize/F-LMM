import sys
from PIL import Image
import warnings
from vlmeval.vlm.base import BaseModel
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.registry import MODELS


class VisualCoTDeepSeekVL(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 config='configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py',
                 checkpoint='',
                 version='v2',
                 **kwargs):
        cfg = Config.fromfile(config)
        self.model = MODELS.build(cfg.model)
        state_dict = guess_load_checkpoint(checkpoint)
        _ = self.model.load_state_dict(state_dict, strict=False)
        self.model._prepare_for_generation(
            image_processor=cfg.image_processor,
            prompt_template=cfg.prompt_template,
            max_thought_tokens=16,
            max_new_tokens=512,
            lmm_name=cfg.lmm_name,
            additional_prompt='',
            with_memory=True,
            use_sam=False,
        )
        self.model = self.model.cuda().eval()
        self.version = version

    @staticmethod
    def prepare_inputs(message):
        content, images = '', []
        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                content += s['value']
        assert len(images) == 1
        image = Image.open(images[0]).convert('RGB')

        return image, content

    def generate_inner(self, message, dataset=None):
        image, content = self.prepare_inputs(message)
        text_output = getattr(self.model, f'visual_cot_{self.version}')(image, content)[-1]
        return text_output
