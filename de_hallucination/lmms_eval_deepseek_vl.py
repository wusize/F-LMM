import torch
import logging
from tqdm import tqdm
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from mmengine.config import Config
from mmengine.registry import MODELS
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union
import warnings
from xtuner.model.utils import guess_load_checkpoint
warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")


class DeepseekVLEval(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        mmengine_config='',
        pretrained: str = "",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        version: str = 'v1',
        with_memory=True,
        use_sam=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        accelerator = Accelerator()
        cfg = Config.fromfile(mmengine_config)
        self._model = MODELS.build(cfg.model)
        state_dict = guess_load_checkpoint(pretrained)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
        self._model._prepare_for_generation(
            image_processor=cfg.image_processor,
            prompt_template=cfg.prompt_template,
            max_thought_tokens=16,
            max_new_tokens=512,
            lmm_name=cfg.lmm_name,
            additional_prompt='',
            with_memory=with_memory,
            use_sam=use_sam,
        )
        self._model.eval()
        self._config = self._model.deepseek_vl.config
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        self.version = version
        self.batch_size_per_gpu = int(batch_size)
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]):
        assert False

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            assert len(visuals) == 1, f"Currently only support 1 image: {contexts}, {visuals}"
            # import pdb; pdb.set_trace()
            if 'max_new_tokens' in gen_kwargs:
                self.model.max_new_tokens = gen_kwargs['max_new_tokens']
            text_output = getattr(self.model, f'visual_cot_{self.version}')(visuals[0], contexts)[-1]
            res.append(text_output)
            pbar.update(1)
        return res
