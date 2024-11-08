from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "Semantic_Memory_Model",
]






# ##########################################
# ### ////// View Encoder Network ////// ###
# ##########################################


class Conv2d(nn.Conv2d): # For Weight Standardization
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = norm_layer(32, planes)
        self.mish = nn.Mish()
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = norm_layer(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.mish(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.gn1 = norm_layer(32, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.gn2 = norm_layer(32, width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.gn3 = norm_layer(32, planes * self.expansion)
        self.mish = nn.Mish()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.mish(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.mish(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = norm_layer(32, self.inplanes)
        self.mish = nn.Mish()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
            

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0003) # init recommended on issues of mish github https://github.com/digantamisra98/Mish/issues/37#issue-744119604 
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last GN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.gn3.weight is not None:
                    nn.init.constant_(m.gn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.gn2.weight is not None:
                    nn.init.constant_(m.gn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(32, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, before_flatten=False, before_avgpool=False) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.mish(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if before_avgpool:
            return x

        x = self.avgpool(x)

        if before_flatten:
            return x
        
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)

    return model


def resnet18(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)

def resnext50_32x4d(**kwargs: Any) ->ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext101_32x8d(**kwargs: Any) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnext101_64x4d(**kwargs: Any) -> ResNet:
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

def wide_resnet50_2(**kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def wide_resnet101_2(**kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)






#############################################
### ////// Semantic Memory Network ////// ###
#############################################

class Semantic_Memory_Model(torch.nn.Module):
    def __init__(self, encoder, num_pseudoclasses, proj_dim=2048):
        super().__init__()
        self.num_pseudoclasses = num_pseudoclasses
        self.proj_dim = proj_dim

        self.encoder = encoder
        self.features_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = torch.nn.Identity()


        #### Projector (R)
        ## Batchnorm
        # self.projector = torch.nn.Sequential(
        #     torch.nn.Linear(self.features_dim, self.proj_dim, bias=False),
        #     nn.BatchNorm1d(self.proj_dim),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(self.proj_dim, self.proj_dim, bias=False),
        #     nn.BatchNorm1d(self.proj_dim),
        #     torch.nn.Mish(),
        # )
        ## GN
        # self.projector = torch.nn.Sequential(
        #     torch.nn.Linear(self.features_dim, self.proj_dim, bias=False),
        #     torch.nn.GroupNorm(32, self.proj_dim),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(self.proj_dim, self.proj_dim, bias=False),
        #     torch.nn.GroupNorm(32, self.proj_dim),
        #     torch.nn.Mish(),
        # )
        ## Batchnorm + WS
        self.projector = torch.nn.Sequential(
            conv1x1(self.features_dim, self.proj_dim), # bias is false. WS
            nn.BatchNorm2d(self.proj_dim),
            torch.nn.Mish(),
            conv1x1(self.proj_dim, self.proj_dim), # bias is false. WS
            nn.BatchNorm2d(self.proj_dim),
            torch.nn.Mish(),
            conv1x1(self.proj_dim, int(self.proj_dim/2)),
        )
        ## GN + WS
        # self.projector = torch.nn.Sequential(
        #     conv1x1(self.features_dim, self.proj_dim), # bias is false. WS
        #     torch.nn.GroupNorm(32, self.proj_dim),
        #     torch.nn.Mish(),
        #     conv1x1(self.proj_dim, self.proj_dim), # bias is false. WS
        #     torch.nn.GroupNorm(32, self.proj_dim),
        #     torch.nn.Mish(),
        #     conv1x1(self.proj_dim, int(self.proj_dim/2)),
        # )


        #### Linear head (F)
        # self.linear_head = torch.nn.Linear(self.proj_dim, self.num_pseudoclasses, bias=True)
        self.linear_head = torch.nn.utils.weight_norm(torch.nn.Linear(int(self.proj_dim/2), self.num_pseudoclasses, bias=False)) # MIRA does this weight normalization
        self.linear_head.weight_g.data.fill_(1)
        self.linear_head.weight_g.requires_grad = False


    def forward(self, x, proj_out=False):

        # forwad when using batchnorm1d or GN in projector
        # x_enc = self.encoder(x)
        # x_proj = self.projector(x_enc)
        # x_proj_norm = F.normalize(x_proj, dim=1) # MIRA needs this normalization
        # x_lin = self.linear_head(x_proj_norm)

        # forward when using GN + WS in projector
        x_enc = self.encoder(x, before_flatten=True)
        x_proj = self.projector(x_enc)
        x_proj = torch.flatten(x_proj, 1)
        x_proj_norm = F.normalize(x_proj, dim=1)
        x_lin = self.linear_head(x_proj_norm)

        if proj_out:
            return x_lin, x_proj
        return x_lin
    

