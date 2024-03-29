import torch.nn as nn
from mmcv.cnn import ConvModule
try:
    from mmseg.models import UNet
except:
    UNet = None


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


class UNetHead(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_seg = nn.Conv2d(self.base_channels, 1, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        x = super().forward(x)
        return self.conv_seg(x[-1])
