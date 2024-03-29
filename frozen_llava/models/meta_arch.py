import torch
from mmengine.model import BaseModel
from xtuner.registry import BUILDER


class FrozenLlava(BaseModel):

    def __init__(self,
                 model,
                 mask_head):
        super().__init__()
        self.llava = BUILDER.build(model)
        self.llava.requires_grad_(False)
        mask_head.update(
            in_channels=self.llava.config.text_config.num_attention_heads*
                        self.llava.config.text_config.num_hidden_layers*2)
        self.mask_head = BUILDER.build(mask_head)
        self.patch_size = self.llava.config.vision_config.patch_size

    def init_weights(self):
        pass

    def train(self, mode=True):
        self.llava.train(mode=False)
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

    def _forward(self, data):
        return None

    def predict(self, data):
        return None

    def compute_loss(self, data):
        import pdb; pdb.set_trace()
        for data_sample in data:
            inputs = dict(input_ids=data_sample['input_ids'].to(self.llava.device),
                          mask_ids=data_sample['mask_ids'].to(self.llava.device),
                          pixel_values=data_sample['pixel_values'].to(device=self.llava.device,
                                                                      dtype=self.llava.dtype),
                          image_sizes=data_sample['image_sizes'].to(self.llava.device))
            with torch.no_grad():
                outputs = self.llava(data_sample)

            masks = inputs['masks'].to(self.llava.device)

        loss_dict = {'loss': torch.tensor(0.0).to(self.llava.device)}
        return loss_dict


    def train_step(self, data, optim_wrapper):
        # Enable automatic mixed precision training context.
        import pdb; pdb.set_trace()
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
