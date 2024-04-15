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
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 in_channels=2048,
                 channels=256,
                 concat_input=True,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__()
        assert num_convs > 0
        self.in_channels = in_channels
        self.channels = channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)
        self.conv_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

    def forward(self, x):
        """Forward function."""
        x = self.convs(x)
        x = self.conv_seg(x)
        return x


def upsample_forward_func(self, x):
    dtype = x.dtype
    x = x.float()
    if not self.size:
        size = [int(t * self.scale_factor) for t in x.shape[-2:]]
    else:
        size = self.size
    return resize(x, size, None, self.mode, self.align_corners).to(dtype)


class UNetHead(UNet):
    def __init__(self, upsample_input=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_seg = nn.Conv2d(self.base_channels, 1, kernel_size=1)

        for module in self.modules():
            if isinstance(module, Upsample):
                print_log("Replace upsample forward function")
                module.forward = types.MethodType(upsample_forward_func, module)

        self.init_weights()
        self.upsample_input = upsample_input

    def forward(self, x):
        h, w = x.shape[-2:]
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


if __name__ == '__main__':
    from mmseg.models.backbones.unet import InterpConv
    unet = UNetHead(in_channels=2048,
                    base_channels=64,
                    num_stages=3,
                    strides=(1, 1, 1),
                    enc_num_convs=(2, 2, 2),  # the first enc is for projection
                    dec_num_convs=(2, 2),
                    downsamples=(True, True),
                    enc_dilations=(1, 1, 1),
                    dec_dilations=(1, 1),
                    upsample_cfg=dict(type=InterpConv))

    torch.save(unet.state_dict(), 'data/unet_example.pth')