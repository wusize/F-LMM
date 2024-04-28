from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mgm.model import MGMGemmaForCausalLM
from mgm.model.processor.video_processor import VideoFramesProcessor


mgm_name = 'YanweiLi/MGM-2B'

model = dict(type=MGMGemmaForCausalLM.from_pretrained,
             pretrained_model_name_or_path=mgm_name,
             mm_vision_tower='openai/clip-vit-large-patch14-336',
             mm_vision_tower_aux='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup',
             torch_dtype=torch.bfloat16,
             low_cpu_mem_usage=True)

tokenizer = dict(type=AutoTokenizer.from_pretrained,
                 pretrained_model_name_or_path=mgm_name)


image_processor = dict(type=VideoFramesProcessor.from_pretrained,
                       pretrained_model_name_or_path=mgm_name)


if __name__ == "__main__":
    from xtuner.registry import BUILDER
    from xtuner.model.utils import LoadWoInit

    model_path = model['pretrained_model_name_or_path']
    with LoadWoInit():
        model = BUILDER.build(model)
    tokenizer = BUILDER.build(tokenizer)
    # image_processor= BUILDER.build(image_processor)

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

