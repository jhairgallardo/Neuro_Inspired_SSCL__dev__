from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import einops
import math
import numpy as np
import matplotlib.pyplot as plt


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
    "ConditionalGenerator",
    "ZCA_layer"
]






# ##########################################
# ### ////// View Encoder Network ////// ###
# ##########################################


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


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
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.act = nn.Mish()
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

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
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = norm_layer(planes * self.expansion)
        self.act = nn.Mish()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

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
        output_before_avgpool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(self.inplanes)
        self.act = nn.Mish()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
            

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last GN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.norm3.weight is not None:
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.norm2.weight is not None:
                    nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

        self.output_before_avgpool = output_before_avgpool

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
                norm_layer(planes * block.expansion),
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

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.output_before_avgpool:
            return x

        x = self.avgpool(x)
        
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


########################################
### ////// ZCA Layer Network ////// ####
########################################

class ZCA_layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, epsilon=1e-4):
        super(ZCA_layer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
        self.act = torch.nn.Tanh()

        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x
    
    def ZCA_function(self, data, epsilon=1e-6):
        ### Function that computes the ZCA matrix
        # data is a d x N matrix, where d is the dimensionality and N is the number of samples
        C = (data @ data.T) / data.size(1)
        D, V = torch.linalg.eigh(C)
        sorted_indices = torch.argsort(D, descending=True)
        D = D[sorted_indices]
        V = V[:, sorted_indices]
        Di = (D + epsilon)**(-0.5)
        zca_matrix = V @ torch.diag(Di) @ V.T
        return zca_matrix
    
    def init_ZCA_layer_weights(self, imgs):

        ### 1) Extract data as Patches
        window_size = self.kernel_size
        step = window_size
        aux = imgs.unfold(2, window_size, step)
        aux = aux.unfold(3, window_size, step)
        patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.in_channels, window_size, window_size)
        data = einops.rearrange(patches, 'n c h w -> (c h w) n') # data is a d x N
        data = data.double() # double precision

        ### 2) Compute ZCA
        W = self.ZCA_function(data, epsilon = self.epsilon) # shape: k (c h w)
        W = einops.rearrange(W, 'k (c h w) -> k c h w', c=self.in_channels, h=self.kernel_size, w=self.kernel_size) # Reshape filters

        ### 3) Pick centered filters as the main ZCA filters
        num_main_filters = self.in_channels # There will be only self.in_channels main zca centered filters
        main_zcafilters = torch.zeros((num_main_filters, self.in_channels, self.kernel_size, self.kernel_size), dtype=W.dtype)
        for i in range(self.in_channels):
            index = int( (self.kernel_size*self.kernel_size)*(i) + (self.kernel_size*self.kernel_size-1)/2 ) # index of the filter that is in the center of the patch
            main_zcafilters[i, :] = W[index, :, :, :]
        main_zcafilters = main_zcafilters.float() # back to single precision
        # plot_filters_and_output(imgs, main_zcafilters, self.in_channels, self.kernel_size, name='mainZCA')

        ### 4) Scale down the filters
        ## Maxscale the filters
        maxscaled_zcafilters = main_zcafilters / torch.amax(main_zcafilters, keepdims=True)
        # plot_filters_and_output(imgs, maxscaled_zcafilters, self.in_channels, self.kernel_size, name='maxscaledZCA')
        ## Get the output of the temporary zca layer
        temp_zcalayer = nn.Conv2d(self.in_channels, maxscaled_zcafilters.shape[0], kernel_size=self.kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
        temp_zcalayer.weight = torch.nn.Parameter(maxscaled_zcafilters)
        temp_zcalayer.eval()
        with torch.no_grad():
            temp_output = temp_zcalayer(imgs)
        ## Get CDF on the absolute values of the output
        temp_output = temp_output.abs().flatten().numpy()
        count, bins_count = np.histogram(temp_output, bins=100) 
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ## Find threshold, multiplier, and scale the filters
        threshold = bins_count[1:][np.argmax(cdf>0.99999)]
        multiplier = np.arctanh(0.999)/threshold # find value where after a tanh function, everythinng after the threshold will be near 1 (0.999)
        scaleddown_zcafilters = maxscaled_zcafilters * multiplier
        # plot_filters_and_output(imgs, scaleddown_zcafilters, self.in_channels, self.kernel_size, name='scaleddownZCA', with_act=True, act=self.act)
        
        ### 5) Set the weights of the ZCA layer
        self.conv.weight = torch.nn.Parameter(scaleddown_zcafilters)
        self.conv.weight.requires_grad = False
        # plot_output_feature_maps(self.conv, self.act, imgs)

        return None
    

### Plot functions to double check the ZCA layer filters and outputs

def plot_filters_and_output(imgs, filters, in_channels, kernel_size, name='mainZCA', with_act=False, act=None):
    for i in range(filters.shape[0]):
        plt.hist(filters[i].flatten().numpy(), bins=10)
        plt.savefig(f'{name}_hist_filter_{i}.png')
        plt.close()
    ### Plot each filter
    for i in range(filters.shape[0]):
        filter_m = filters[i]
        filter_m_norm = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min())
        filter_m_norm = filter_m_norm.cpu().numpy().transpose(1,2,0)
        plt.imshow(filter_m_norm)
        plt.axis('off')
        plt.savefig(f'{name}_filter_{i}.png', bbox_inches='tight')
        plt.close()
    ### Plot layer output
    auxconv = nn.Conv2d(in_channels, filters.shape[0], kernel_size=kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
    auxconv.weight = torch.nn.Parameter(filters)
    auxconv.eval()
    with torch.no_grad():
        auxoutput = auxconv(imgs)
    auxoutput = auxoutput.flatten().numpy()
    plt.figure(figsize=(18,6))
    plt.subplot(2,1,1)
    plt.hist(auxoutput, bins=100)
    plt.ylabel('frequency')
    plt.title(f'{name} layer output')
    plt.subplot(2,1,2)
    plt.hist(auxoutput, bins=100)
    plt.yscale('log')
    plt.ylabel('log scale frequency')
    plt.savefig(f'{name}_output.png', bbox_inches='tight')
    plt.close()

    if with_act:
        with torch.no_grad():
            auxoutput = act(auxconv(imgs))
        auxoutput = auxoutput.flatten().numpy()
        plt.figure(figsize=(18,6))
        plt.subplot(2,1,1)
        plt.hist(auxoutput, bins=100)
        plt.ylabel('frequency')
        plt.title(f'{name} layer output with activation')
        plt.subplot(2,1,2)
        plt.hist(auxoutput, bins=100)
        plt.yscale('log')
        plt.ylabel('log scale frequency')
        plt.savefig(f'{name}_output_with_act.png', bbox_inches='tight')
        plt.close()
    return None

def plot_output_feature_maps(conv, act, imgs):
    with torch.no_grad():
        output = conv(imgs)
        output_act = act(output)

    ### plot output images (plot output as a 3 channel image)
    # ZCA layer output image
    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    for i in range(16):
        output_img = output[i].permute(1,2,0)
        output_img = output_img - torch.min(output_img)
        output_img = output_img / torch.max(output_img)
        output_img = output_img.numpy()
        ax = axs[i//4, i%4]
        ax.imshow(output_img)
        ax.axis('off')
        ax.set_title(f'output image {i}')
    plt.savefig('output_images.png', bbox_inches='tight')
    plt.close()
    # ZCA layer output image with activation
    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    for i in range(16):
        output_img = output_act[i].permute(1,2,0)
        output_img = output_img - torch.min(output_img)
        output_img = output_img / torch.max(output_img)
        output_img = output_img.numpy()
        ax = axs[i//4, i%4]
        ax.imshow(output_img)
        ax.axis('off')
        ax.set_title(f'output image {i} with activation')
    plt.savefig('output_images_with_act.png', bbox_inches='tight')
    plt.close()
    
    ### plot output feature maps (absolute values the mean across channels)
    output_abs_mean = output.abs().mean(dim=1).numpy() # shape: N x H x W
    output_act_abs_mean = output_act.abs().mean(dim=1).numpy() # shape: N x H x W
    # ZCA layer output
    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    for i in range(16):
        ax = axs[i//4, i%4]
        ax.imshow(output_abs_mean[i])
        ax.axis('off')
        ax.set_title(f'output image {i}')
    plt.savefig('output_absmean_images.png', bbox_inches='tight')
    plt.close()
    # ZCA layer output with activation
    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    for i in range(16):
        ax = axs[i//4, i%4]
        ax.imshow(output_act_abs_mean[i])
        ax.axis('off')
        ax.set_title(f'output image {i} with activation')
    plt.savefig('output_absmean_images_with_act.png', bbox_inches='tight')
    plt.close()
    return None



###################################################
### ////// Conditional Generator Network ////// ###
###################################################

### ////// FeatureTransformation Network ////// ###

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model, 
                 max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class FeatureTransformationNetwork(nn.Module):
    def __init__(self, 
                 feature_dim=512, 
                 action_code_dim=11, 
                 num_layers=6, 
                 nhead=8, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super(FeatureTransformationNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.action_code_dim = action_code_dim
        self.sequence_length = 50  # 1 action token + 49 feature tokens (7x7)

        # MLP to transform the action code into a 512-length token
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_code_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        # MLP to map features tokens into a space of the same dimension. 
        # This can help to have it in the same space as the action code
        self.feature_mlp = nn.Linear(self.feature_dim, self.feature_dim)

        # Positional Encoding for the sequence
        self.positional_encoding = PositionalEncoding(d_model=self.feature_dim, max_len=self.sequence_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.act = nn.Mish()

    def forward(self, feature_map, action_code):
        """
        action_code: Tensor of shape (batch_size, 11)
        feature_map: Tensor of shape (batch_size, 512, 7, 7)
        Returns:
            transformed_feature_map: Tensor of shape (batch_size, 512, 7, 7)
        """
        batch_size = action_code.size(0)

        # Step 1: Transform action code into a 512-length token
        action_token = self.action_mlp(action_code)  # shape: (batch_size, 512)
        action_token = action_token.unsqueeze(1)  # shape: (batch_size, 1, 512)

        # Step 2: Flatten the feature map into 49 tokens of length 512 and apply a linear layer
        feature_tokens = einops.rearrange(feature_map, 'b c h w -> b (h w) c')  # shape: (batch_size, 49, 512)
        feature_tokens = self.feature_mlp(feature_tokens)

        # Step 3: Concatenate the action token with feature tokens (action token first)
        tokens = torch.cat((action_token, feature_tokens), dim=1)  # shape: (batch_size, 50, 512)

        # Step 4: Add positional encoding
        tokens = self.positional_encoding(tokens)

        # Step 5: Prepare tokens for Transformer (sequence length first)
        tokens = tokens.permute(1, 0, 2)  # shape: (50, batch_size, 512)

        # Step 6: Pass through the Transformer Encoder
        transformed_tokens = self.transformer_encoder(tokens)  # shape: (50, batch_size, 512)

        # Step 7: Drop the first token (action code) and reshape
        transformed_feature_tokens = transformed_tokens[1:, :, :].permute(1, 2, 0)  # shape: (batch_size, 512, 49)
        transformed_feature_map = transformed_feature_tokens.view(batch_size, self.feature_dim, 7, 7)  # shape: (batch_size,512, 7, 7)
        transformed_feature_map = self.act(transformed_feature_map)

        return transformed_feature_map

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
    
class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)
        self.norm2 = nn.GroupNorm(min([32, in_planes//4]), in_planes)
        self.act = nn.Mish()
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)
            self.norm1 = nn.GroupNorm(min([32, in_planes//4]), planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='replicate')
            self.norm1 = nn.GroupNorm(min([32, in_planes//4]), planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='replicate'),
            )
    def forward(self, x):
        out = self.act(self.norm2(self.conv2(x)))
        out = self.norm1(self.conv1(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNetDec(nn.Module):
    def __init__(self, 
                 num_Blocks=[1,1,1,1], 
                 nc=3):
        super().__init__()
        self.in_planes = 512
        self.out_act = nn.Tanh() # Because we are reconstructing input images with values between -1 and 1

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)

        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockDec(self.in_planes, planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.out_act(self.conv1(x))
        return x
    

### ////// Conditional Generator Network ////// ###

class ConditionalGenerator(nn.Module):
    def __init__(self,
                 dec_num_Blocks = [1,1,1,1],
                 dec_num_out_channels = 3,
                 ft_feature_dim=512, 
                 ft_action_code_dim=11, 
                 ft_num_layers=2, 
                 ft_nhead=4, 
                 ft_dim_feedforward=256, 
                 ft_dropout=0.1):
        super(ConditionalGenerator, self).__init__()

        # Define feature transformation network
        self.feature_transformation_network = FeatureTransformationNetwork(feature_dim=ft_feature_dim, 
                                                                           action_code_dim=ft_action_code_dim, 
                                                                           num_layers=ft_num_layers, 
                                                                           nhead=ft_nhead, 
                                                                           dim_feedforward=ft_dim_feedforward, 
                                                                           dropout=ft_dropout)

        # Define decoder
        self.decoder = ResNetDec(num_Blocks=dec_num_Blocks, nc=dec_num_out_channels)



    def forward(self, feature_map, action_code, skip_FTN=False):
        if skip_FTN: 
            # Pass tensor direcly to the decoder to boost learning (It helps training the generator to create better images)
            generated_image = self.decoder(feature_map)
            return generated_image
        else: 
            # Use the feature transformation network to learn to predict the next view tensor using action code and previous view.
            # This option also works when action code means "no action" (i.e., the next view is the same as the previous view)
            transformed_feature_map = self.feature_transformation_network(feature_map, action_code)
            generated_image = self.decoder(transformed_feature_map)
            return generated_image, transformed_feature_map
