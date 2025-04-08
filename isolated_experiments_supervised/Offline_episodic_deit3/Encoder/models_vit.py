from torch import nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer

__all__ = [
    'vit_tiny_patch16_224',
]


def vit_tiny_patch16_224(img_size=224, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model = VisionTransformer(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

### Test models in main
if __name__ == '__main__':
    import torch
    from torchinfo import summary

    model = vit_tiny_patch16_224(num_classes=1000)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    batch_size = 16
    summary(model, input_size=(batch_size, 3, 224, 224), device='cpu')