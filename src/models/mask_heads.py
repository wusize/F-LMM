import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models import UNet
from mmseg.models.utils.wrappers import Upsample, resize
from mmengine.logging import print_log


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 in_channels=2048,
                 channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 upsample_input=None,
                 normalize_input=False,):
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        super().__init__()
        assert num_convs > 0
        self.in_channels = in_channels
        self.channels = channels

        conv_padding = kernel_size // 2
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        )
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )
        self.convs = nn.Sequential(*convs)
        self.conv_seg = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.upsample_input = upsample_input
        self.normalize_input = normalize_input

    def forward(self, x):
        """Forward function."""
        if self.normalize_input:
            assert x.min() >= 0.0 and x.max() <= 1.0
            x_sum = x.sum((-2, -1), keepdims=True).clamp(min=1e-12)
            x = x / x_sum
        if self.upsample_input is not None:
            h, w = x.shape[-2:]
            scale_factor = max(1.0, self.upsample_input / max(h, w))
            x = F.interpolate(x.float(), scale_factor=scale_factor, mode='bilinear').to(x)
        x = self.convs(x)
        x = self.conv_seg(x)
        return x

    @property
    def dtype(self):
        return self.conv_seg.weight.dtype


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


class SingleConvHead(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 upsample_input=64,
                 kernel_size=5,   # determine receptive field
                 normalize_input=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1,
                              kernel_size=kernel_size,
                              stride=1, padding=(kernel_size - 1) // 2)
        self.upsample_input = upsample_input
        self.normalize_input = normalize_input

    def forward(self, x):
        if self.normalize_input:
            assert x.min() >= 0.0 and x.max() <= 1.0
            x_sum = x.sum((-2, -1), keepdims=True).clamp(min=1e-12)
            x = x / x_sum
        h, w = x.shape[-2:]
        if self.upsample_input is not None:
            scale_factor = max(1.0, self.upsample_input / max(h, w))
            x = F.interpolate(x.float(), scale_factor=scale_factor, mode='bilinear').to(x)
            h, w = x.shape[-2:]
        x = self.conv(x)
        assert x.shape[-2:] == (h, w)
        return x

    @property
    def dtype(self):
        return self.conv.weight.dtype


if __name__ == '__main__':
    from mmseg.models.backbones.unet import InterpConv
    from torch.nn.modules.normalization import GroupNorm
    import numpy as np
    unet = UNetHead(normalize_input=True,
                    upsample_input=64,   # upsample the low-res input (24x24) to (64 x 64)
                    in_channels=2048,
                    base_channels=64,
                    num_stages=4,strides=(1, 1, 1, 1),
                    enc_num_convs=(2, 2, 2, 2),   # the first enc is for projection
                    dec_num_convs=(2, 2, 2),
                    downsamples=(True, True, True),
                    enc_dilations=(1, 1, 1, 1),
                    dec_dilations=(1, 1, 1),
                    norm_cfg=dict(type=GroupNorm, num_groups=1),
                    upsample_cfg=dict(type=InterpConv)
                    )

    n_params = sum([np.prod(p.size()) for p in unet.parameters() if p.requires_grad])
    print(n_params)

    torch.save(unet.state_dict(), 'data/unet.pth')

    fcn = FCNHead(normalize_input=True,
                  upsample_input=64,
                  num_convs=8,
                  kernel_size=3,
                  in_channels=2048,
                  norm_cfg=dict(type=GroupNorm, num_groups=1),
                  channels=256)
    n_params = sum([np.prod(p.size()) for p in fcn.parameters() if p.requires_grad])
    print(n_params)
    torch.save(fcn.state_dict(), 'data/fcn.pth')
