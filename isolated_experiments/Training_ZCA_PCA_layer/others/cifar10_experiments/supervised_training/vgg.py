from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(
        self, cfg_str, batchnorm_flag=True, num_classes=10, init_weights=True, conv0_flag=False, conv0_outchannels=10
    ) -> None:
        super().__init__()

        self.conv0_flag = conv0_flag
        self.conv0_outchannels = conv0_outchannels
        if conv0_flag:
            self.conv0 = nn.Conv2d(3, conv0_outchannels, kernel_size=3, stride=1, padding=1, bias=True)
            self.act0 = nn.GELU()

        self.features = self._make_layers(cfgs[cfg_str], batchnorm_flag)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.conv0_flag:
            x = self.conv0(x)
            x = self.act0(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg: List[Union[str, int]], batchnorm_flag: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []

        if self.conv0_flag:
            in_channels = self.conv0_outchannels
        else:
            in_channels = 3

        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batchnorm_flag:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.GELU()]
                else:
                    layers += [conv2d, nn.GELU()]
                in_channels = v
        return nn.Sequential(*layers)


def vgg11(**kwargs: Any) -> VGG:
    return VGG("A", batchnorm_flag=False, **kwargs)

def vgg11_bn(**kwargs: Any) -> VGG:
    return VGG("A", batchnorm_flag=True, **kwargs)

def vgg13(**kwargs: Any) -> VGG:
    return VGG("B", batchnorm_flag= False, **kwargs)

def vgg13_bn(**kwargs: Any) -> VGG:
    return VGG("B", batchnorm_flag=True, **kwargs)

def vgg16(**kwargs: Any) -> VGG:
    return VGG("D", batchnorm_flag=False, **kwargs)

def vgg16_bn(**kwargs: Any) -> VGG:
    return VGG("D", batchnorm_flag=True, **kwargs)

def vgg19(**kwargs: Any) -> VGG:
    return VGG("E", batchnorm_flag=False, **kwargs)

def vgg19_bn(**kwargs: Any) -> VGG:
    return VGG("E", batchnorm_flag=True, **kwargs)