import math
import types
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import UNet
from mmseg.models.utils.wrappers import Upsample, resize
from mmengine.logging import print_log


def upsample_forward_func(self, x):
    dtype = x.dtype
    x = x.float()
    if not self.size:
        size = [int(t * self.scale_factor) for t in x.shape[-2:]]
    else:
        size = self.size
    return resize(x, size, None, self.mode, self.align_corners).to(dtype)


class UNetHead(UNet):
    def __init__(self, upsample_input=None,
                 normalize_input=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_seg = nn.Conv2d(self.base_channels, 1, kernel_size=1)

        for module in self.modules():
            if isinstance(module, Upsample):
                print_log("Replace upsample forward function")
                module.forward = types.MethodType(upsample_forward_func, module)

        self.init_weights()
        self.upsample_input = upsample_input
        self.normalize_input = normalize_input

    @property
    def dtype(self):
        return self.conv_seg.weight.dtype

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.normalize_input:
            assert x.min() >= 0.0 and x.max() <= 1.0
            x_sum = x.sum((-2, -1), keepdims=True).clamp(min=1e-12)
            x = x / x_sum

        if self.upsample_input is not None:
            scale_factor = max(1.0, self.upsample_input / max(h, w))
            x = F.interpolate(x.float(), scale_factor=scale_factor, mode='bilinear').to(x)
            h, w = x.shape[-2:]   # upsample the low-res input to get better results

        dividend = 2**(self.num_stages - 1)
        padded_h = math.ceil(h / dividend) * dividend
        padded_w = math.ceil(w / dividend) * dividend

        padded_x = x.new_zeros(*x.shape[:2], padded_h, padded_w)
        padded_x[..., :h, :w] = x
        x = super().forward(padded_x)[-1][..., :h, :w]
        return self.conv_seg(x)
