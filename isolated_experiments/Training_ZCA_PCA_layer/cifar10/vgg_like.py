import torch
import torch.nn as nn

from typing import Any

__all__ = [
    "vgglike",
]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

class VGGlike_class(nn.Module):
    def __init__(
        self,
        widths={
                'block1': 64,
                'block2': 256,
                'block3': 256,
                }, 
        batchnorm_momentum=0.6,
        num_classes: int = 1000,
        conv0_flag=False, 
        conv0_outchannels=10,
        conv0_kernel_size=3,
    ) -> None:
        super().__init__()
        whiten_kernel_size = conv0_kernel_size
        whiten_width = conv0_outchannels

        self.conv0_flag = conv0_flag
        if self.conv0_flag:
            self.conv0 = nn.Conv2d(3, whiten_width, kernel_size=whiten_kernel_size, stride=1, padding=0, bias=True)
            self.act0 = nn.GELU()
            self.conv_group1 = ConvGroup(whiten_width, widths['block1'], batchnorm_momentum)
        else:
            self.conv_group1 = ConvGroup(3, widths['block1'], batchnorm_momentum)

        self.conv_group2 = ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum)
        self.conv_group3 = ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum)
        self.pool = nn.MaxPool2d(3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths['block3'], num_classes, bias=False)
        self.mul = Mul(1/9)

    def forward(self, x):
        if self.conv0_flag:
            x = self.conv0(x)
            x = self.act0(x)
        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)
        x = self.pool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.mul(x)
        return x
    
def vgglike(**kwargs: Any) -> VGGlike_class:
    return VGGlike_class(**kwargs)