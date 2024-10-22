from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class FeatureTransModelNew(nn.Module):
    def __init__(self, action_dim=4, action_channels=32, layer=4):
        super(FeatureTransModelNew, self).__init__()
        self.film = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 49 * action_channels)
        )
        self.action_channels = action_channels
        self.layer = layer

        self.blocks = nn.ModuleList()

        for i in range(layer):
            if i == 0:
                # First block uses kernel_size=1 for conv1
                conv1 = nn.Conv2d(512 + action_channels, 512, kernel_size=1)
            else:
                conv1 = nn.Conv2d(512 + action_channels, 512, kernel_size=3, padding=1)
            bn1 = nn.BatchNorm2d(512)
            conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            bn2 = nn.BatchNorm2d(512)

            block = nn.ModuleDict({
                'conv1': conv1,
                'bn1': bn1,
                'conv2': conv2,
                'bn2': bn2
            })
            self.blocks.append(block)

        self.act = nn.ReLU()

    def forward(self, x, bb):
        # Compute FiLM
        film = self.film(bb).view(-1, self.action_channels, 7, 7)
        identical = x

        out = x
        for i, block in enumerate(self.blocks):
            identity = out if i > 0 else identical

            out = torch.cat([out, film], dim=1)
            out = block['conv1'](out)
            out = block['bn1'](out)
            out = self.act(out)
            out = block['conv2'](out)
            out = block['bn2'](out) + identity

            # Apply activation function except after the last block
            if i < self.layer - 1:
                out = self.act(out)

        return out

### The following code is the same with unconditional generator

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        conv0_flag=False,
        conv0_outchannels=6,
        conv0_kernel_size=3,
        act0 = nn.Mish(),
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

        self.conv0_flag = conv0_flag
        if self.conv0_flag:
            #self.conv0 = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', bias=False) # check bias
            self.conv0 = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
            self.act0 = act0 #nn.Mish()
            self.conv1 = nn.Conv2d(conv0_outchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.gn1 = norm_layer(32, self.inplanes)
        self.mish = nn.Mish()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # standard resnet init
                # nn.init.kaiming_normal_(m.weight) # init on mish paper https://github.com/digantamisra98/Mish/blob/a60f40a0f8cc8f95c79cf13cc742e5783e548215/exps/resnet.py#L10C5-L10C18
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        if self.conv0_flag:
            x = self.conv0(x)
            x = self.act0(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.mish(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


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



### ////// Decoder Network ////// ###
## Deconvolution and checkerboard artifacts
## https://distill.pub/2016/deconv-checkerboard/
## mode='nearest', 'bilinear', 'bicubic'

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, padding=1, padding_mode='zeros', mode='bicubic'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False, padding_mode=padding_mode)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, beta, gamma):
        if 0 in beta.size() and 0 in gamma.size():
            return x
        else:
            beta = beta.view(x.size(0), x.size(1), 1, 1)
            gamma = gamma.view(x.size(0), x.size(1), 1, 1)
            x = gamma * x + beta
            return x

class FiLM2DBlock(nn.Module): # spatial film
    def __init__(self):
        super(FiLM2DBlock, self).__init__()

    def forward(self, x, beta, gamma):
        x = gamma * x + beta
        return x



class FilmedBasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        if in_planes%32 == 0:
            self.gn2 = nn.GroupNorm(32, in_planes)
        else:
            self.gn2 = nn.GroupNorm(1, in_planes)

        self.mish = nn.Mish()
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(32, planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.gn1 = nn.GroupNorm(32, planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
            )

        self.film = FiLMBlock()

    def forward(self, x, beta, gamma):
        out = self.conv2(x)
        out = self.gn2(out)
        # add film here firstly. check if it works.
        out = self.film(out, beta, gamma)

        out = self.mish(out)
        out = self.conv1(out)

        out = self.gn1(out)
        out += self.shortcut(x)
        out = self.mish(out)
        return out

def create_film_generator(feature_dim, film_type='linear',condition_dim=4,final_layer_scale=1):
    if sum(feature_dim) == 0:
        return nn.Identity(), nn.Identity()
    if film_type == 'linear':
        film_generator_gamma = nn.Linear(condition_dim, sum(feature_dim))
        film_generator_beta = nn.Linear(condition_dim, sum(feature_dim))

        # gamma around 1, beta around 0
        film_generator_gamma.weight = nn.Parameter(film_generator_gamma.weight / final_layer_scale)
        film_generator_gamma.bias = nn.Parameter(film_generator_gamma.bias / final_layer_scale + 1)

        film_generator_beta.weight = nn.Parameter(film_generator_beta.weight / final_layer_scale)
        film_generator_beta.bias = nn.Parameter(film_generator_beta.bias / final_layer_scale)
    elif film_type == 'mlp':
        film_generator_gamma = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, sum(feature_dim))
        )
        film_generator_beta = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, sum(feature_dim))
        )
        # gamma around 1, beta around 0
        film_generator_gamma[2].weight = nn.Parameter(film_generator_gamma[2].weight / final_layer_scale)
        film_generator_gamma[2].bias = nn.Parameter(film_generator_gamma[2].bias / final_layer_scale + 1)
        film_generator_beta[2].weight = nn.Parameter(film_generator_beta[2].weight / final_layer_scale)
        film_generator_beta[2].bias = nn.Parameter(film_generator_beta[2].bias / final_layer_scale)
    else:
        raise NotImplementedError
    return film_generator_beta, film_generator_gamma


# todo: make film 2d
class FilmedResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3, use_coordconv=False, film_type='linear', film_scale=1, first_film_only=False):
        super().__init__()
        assert film_type in ('linear', 'mlp'), "unsupported type of film"

        # calculate film_dim on the fly
        self.film_dims = []
        self.in_planes = 512 if not use_coordconv else 514
        self.use_coordconv = use_coordconv
        self.act = nn.Tanh() # optimal
        # why stride 2 here?
        self.layers = []
        self._make_layer(FilmedBasicBlockDec, 256, num_Blocks[3], stride=2)
        self._make_layer(FilmedBasicBlockDec, 128, num_Blocks[2], stride=2)
        self._make_layer(FilmedBasicBlockDec, 64, num_Blocks[1], stride=2)
        self._make_layer(FilmedBasicBlockDec, 64, num_Blocks[0], stride=2)
        self.layers = nn.ModuleList(self.layers)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size
        ## Embedding vector to condition generation
        # use a linear layer first; change if not working
        if first_film_only:
            self.film_dims = [self.film_dims[0]] + [0 for _ in range(1, len(self.film_dims))]
        self.film_generator_beta, self.film_generator_gamma = create_film_generator(self.film_dims, film_type=film_type, final_layer_scale=film_scale)


        print(f"Film dims: {self.film_dims}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0003) # init recommended on issues of mish github https://github.com/digantamisra98/Mish/issues/37#issue-744119604
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        # for stride in reversed(strides):
        for stride in strides:
            self.layers.append(BasicBlockDec(self.in_planes, planes, stride))
            self.film_dims.append(self.in_planes)
            self.in_planes = planes
        return

    def forward(self, x, condition):
        film_betas = self.film_generator_beta(condition)
        film_gammas = self.film_generator_gamma(condition)
        # keep the number of channels at the end dim
        betas = torch.split(film_betas, self.film_dims, dim=-1)
        gammas = torch.split(film_gammas, self.film_dims, dim=-1)
        if self.use_coordconv:
            n, w, h = x.size(0), x.size(2), x.size(3)
            # dirty fix
            target_device = x.device
            coord_x = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, 1, w, h).to(target_device)
            coord_y = torch.linspace(-1, 1, h).view(1, 1, 1, h).expand(n, 1, w, h).to(target_device)
            x = torch.cat([x, coord_x, coord_y], dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x, betas[i], gammas[i])
        x = self.act(self.conv1(x))

        return x


# for debug only
class PostFilmedResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3, use_coordconv=False, film_type='linear', film_scale=1, first_film_only=False):
        super().__init__()
        assert film_type in ('linear', 'mlp'), "unsupported type of film"

        # calculate film_dim on the fly
        self.film_dims = []
        self.in_planes = 512 if not use_coordconv else 514
        self.use_coordconv = use_coordconv
        self.act = nn.Tanh() # optimal
        # why stride 2 here?
        self.layers = []
        self._make_layer(FilmedBasicBlockDec, 256, num_Blocks[3], stride=2)
        self._make_layer(FilmedBasicBlockDec, 128, num_Blocks[2], stride=2)
        self._make_layer(FilmedBasicBlockDec, 64, num_Blocks[1], stride=2)
        self._make_layer(FilmedBasicBlockDec, 64, num_Blocks[0], stride=2)
        self.layers = nn.ModuleList(self.layers)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size
        ## Embedding vector to condition generation
        # use a linear layer first; change if not working
        # if first_film_only:
        #     self.film_dims = [self.film_dims[0]] + [0 for _ in range(1, len(self.film_dims))]
        # self.film_dims = [0 for _ in range(len(self.film_dims))]
        # self.film_generator_beta, self.film_generator_gamma = create_film_generator(self.film_dims, film_type=film_type, final_layer_scale=film_scale)


        print(f"Film dims: {self.film_dims}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0003) # init recommended on issues of mish github https://github.com/digantamisra98/Mish/issues/37#issue-744119604
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        # for stride in reversed(strides):
        for stride in strides:
            self.layers.append(BasicBlockDec(self.in_planes, planes, stride))
            self.film_dims.append(self.in_planes)
            self.in_planes = planes
        return

    def forward(self, x, condition):
        target_device = x.device
        betas = torch.zeros(x.size(0), 0).to(target_device)
        gammas = torch.zeros(x.size(0), 0).to(target_device)

        # film_betas = self.film_generator_beta(condition)
        # film_gammas = self.film_generator_gamma(condition)
        # keep the number of channels at the end dim
#         betas = torch.split(film_betas, self.film_dims, dim=-1)
#         gammas = torch.split(film_gammas, self.film_dims, dim=-1)
        if self.use_coordconv:
            n, w, h = x.size(0), x.size(2), x.size(3)
            # dirty fix
            target_device = x.device
            coord_x = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, 1, w, h).to(target_device)
            coord_y = torch.linspace(-1, 1, h).view(1, 1, 1, h).expand(n, 1, w, h).to(target_device)
            x = torch.cat([x, coord_x, coord_y], dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x, betas[i], gammas[i])
        x = self.act(self.conv1(x))
        x_cropped = []
        for cond, img in zip(condition, x):
            i,j,h,w = cond
            i = int(i)
            j = int(j)
            h = int(h)
            w = int(w)
            cropped = img[:,i:i+h,j:j+w]
            x_cropped.append(torch.nn.functional.interpolate(cropped.unsqueeze(0), size=(224,224), mode='bilinear').squeeze(0))
        x_cropped = torch.stack(x_cropped)


        return x, x_cropped



class DummyFilmedBasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        if in_planes%32 == 0:
            self.gn2 = nn.GroupNorm(32, in_planes)
        else:
            self.gn2 = nn.GroupNorm(1, in_planes)

        self.mish = nn.Mish()
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(32, planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.gn1 = nn.GroupNorm(32, planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
            )

    def forward(self, x):
        out = self.conv2(x)
        out = self.gn2(out)
        # add film here firstly. check if it works.
        # out = self.film(out, beta, gamma)

        out = self.mish(out)
        out = self.conv1(out)

        out = self.gn1(out)
        out += self.shortcut(x)
        out = self.mish(out)
        return out



class DummyFilmedResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        # calculate film_dim on the fly
        self.film_dims = []
        self.in_planes = 512
        self.act = nn.Tanh() # optimal

        self.layers = []
        self._make_layer(DummyFilmedBasicBlockDec, 256, num_Blocks[3], stride=2)
        self._make_layer(DummyFilmedBasicBlockDec, 128, num_Blocks[2], stride=2)
        self._make_layer(DummyFilmedBasicBlockDec, 64, num_Blocks[1], stride=2)
        self._make_layer(DummyFilmedBasicBlockDec, 64, num_Blocks[0], stride=2)
        self.layers = nn.ModuleList(self.layers)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size
        ## Embedding vector to condition generation
        # use a linear layer first; change if not working
        print(f"Film dims: {self.film_dims}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0003) # init recommended on issues of mish github https://github.com/digantamisra98/Mish/issues/37#issue-744119604
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        # for stride in reversed(strides):
        for stride in strides:
            self.layers.append(BasicBlockDec(self.in_planes, planes, stride))
            self.film_dims.append(self.in_planes)
            self.in_planes = planes
        return

    def forward(self, x, condition):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.act(self.conv1(x))

        return x




class PostFilmedResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3, use_coordconv=False, film_type='linear', film_scale=1, first_film_only=False):
        super().__init__()
        assert film_type in ('linear', 'mlp'), "unsupported type of film"

        # calculate film_dim on the fly
        self.film_dims = []
        self.in_planes = 512 if not use_coordconv else 514
        self.use_coordconv = use_coordconv
        self.act = nn.Tanh() # optimal
        # why stride 2 here?
        self.layers = []
        self._make_layer(FilmedBasicBlockDec, 256, num_Blocks[3], stride=2)
        self._make_layer(FilmedBasicBlockDec, 128, num_Blocks[2], stride=2)
        self._make_layer(FilmedBasicBlockDec, 64, num_Blocks[1], stride=2)
        self._make_layer(FilmedBasicBlockDec, 64, num_Blocks[0], stride=2)
        self.layers = nn.ModuleList(self.layers)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size
        ## Embedding vector to condition generation
        # use a linear layer first; change if not working
        # if first_film_only:
        #     self.film_dims = [self.film_dims[0]] + [0 for _ in range(1, len(self.film_dims))]
        # self.film_dims = [0 for _ in range(len(self.film_dims))]
        # self.film_generator_beta, self.film_generator_gamma = create_film_generator(self.film_dims, film_type=film_type, final_layer_scale=film_scale)


        print(f"Film dims: {self.film_dims}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0003) # init recommended on issues of mish github https://github.com/digantamisra98/Mish/issues/37#issue-744119604
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        # for stride in reversed(strides):
        for stride in strides:
            self.layers.append(BasicBlockDec(self.in_planes, planes, stride))
            self.film_dims.append(self.in_planes)
            self.in_planes = planes
        return

    def forward(self, x, condition):
        target_device = x.device
        betas = torch.zeros(x.size(0), 0).to(target_device)
        gammas = torch.zeros(x.size(0), 0).to(target_device)

        # film_betas = self.film_generator_beta(condition)
        # film_gammas = self.film_generator_gamma(condition)
        # keep the number of channels at the end dim
#         betas = torch.split(film_betas, self.film_dims, dim=-1)
#         gammas = torch.split(film_gammas, self.film_dims, dim=-1)
        if self.use_coordconv:
            n, w, h = x.size(0), x.size(2), x.size(3)
            # dirty fix
            target_device = x.device
            coord_x = torch.linspace(-1, 1, w).view(1, 1, w, 1).expand(n, 1, w, h).to(target_device)
            coord_y = torch.linspace(-1, 1, h).view(1, 1, 1, h).expand(n, 1, w, h).to(target_device)
            x = torch.cat([x, coord_x, coord_y], dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x, betas[i], gammas[i])
        x = self.act(self.conv1(x))
        x_cropped = []
        for cond, img in zip(condition, x):
            i,j,h,w = cond
            i = int(i)
            j = int(j)
            h = int(h)
            w = int(w)
            cropped = img[:,i:i+h,j:j+w]
            x_cropped.append(torch.nn.functional.interpolate(cropped.unsqueeze(0), size=(224,224), mode='bilinear').squeeze(0))
        x_cropped = torch.stack(x_cropped)


        return x, x_cropped







class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, in_planes)
        self.mish = nn.Mish()
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(32, planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.gn1 = nn.GroupNorm(32, planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
            )
    def forward(self, x):
        out = self.mish(self.gn2(self.conv2(x)))
        #print(out.shape)
        out = self.gn1(self.conv1(out))
        #print(out.shape)
        out += self.shortcut(x)
        #print(out.shape)
        out = self.mish(out)
        return out


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[1,1,1,1], nc=3):
        super().__init__()
        self.in_planes = 512
        self.act = nn.Tanh() # optimal for RGB recon
        #self.act = nn.Sequential(nn.Mish(), nn.Tanh()) # optimal for ZCA recon

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)

        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size
        #self.conv1 = ResizeConv2d(64, nc, kernel_size=5, scale_factor=2, padding=2, padding_mode='replicate') ## 5x5 kernel size
        #self.conv1 = ResizeConv2d(64, nc, kernel_size=7, scale_factor=2, padding=3, padding_mode='replicate') ## 7x7 kernel size


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # standard resnet init
                # nn.init.kaiming_normal_(m.weight) # init on mish paper https://github.com/digantamisra98/Mish/blob/a60f40a0f8cc8f95c79cf13cc742e5783e548215/exps/resnet.py#L10C5-L10C18
                nn.init.kaiming_uniform_(m.weight, a=0.0003) # init recommended on issues of mish github https://github.com/digantamisra98/Mish/issues/37#issue-744119604
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockDec(self.in_planes, planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)


    def forward(self, x):
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.act(self.conv1(x))
        #x = self.conv1(x) ## No Activation
        #print(x.shape)
        return x









if __name__ == "__main__":
    from types import SimpleNamespace

    encoder = resnet18()

    x = torch.randn((96, 3, 224, 224), dtype=torch.float32)
    b = torch.ones(96, 4)

    z = encoder(x)
    print("Shape of z:", z.shape)

    decoder = FilmedResNet18Dec(num_Blocks=[1,1,1,1])
    recon = decoder(z, b)

    print('Shape of recon:', recon.shape)

    n_parameters = sum(p.numel() for p in decoder.parameters())
    print('\nNumber of Params (in Millions):', n_parameters / 1e6)