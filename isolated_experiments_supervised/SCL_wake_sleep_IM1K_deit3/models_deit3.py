# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    'deit_tiny_patch16_LS',
    'deit_small_patch16_LS',
    'deit_medium_patch16_LS',
    'deit_base_patch16_LS',
    'deit_large_patch16_LS',
    'Classifier_Model'
]

# ##########################################
# ### ////// View Encoder Network ////// ###
# ##########################################

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,
                output_before_pool=False,
                **kwargs):
        super().__init__()
        self.output_before_pool = output_before_pool

        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):

        # Patchify, add pos embed and cls token
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = x + self.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply blocks
        for i , blk in enumerate(self.blocks):
            x = blk(x)
        # Normalize final output (if pooling is avg, do normalization after pooling)
        x = self.norm(x)

        if self.output_before_pool:
            return x

        # pooling is 'token' (take cls token)
        x = x[:, 0]        

        # drop out before classifier head
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        # Classifier head
        x = self.head(x)

        return x

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

def deit_tiny_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model
    
def deit_small_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model

def deit_medium_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model 

def deit_base_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model
    
def deit_large_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model


########################################
### ////// Classifier Network ////// ###
########################################

class Classifier_Model(torch.nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()

        #### Classifier_head
        self.classifier_head = torch.nn.Linear(input_dim, num_classes) 

    def forward(self, x):
        x = self.classifier_head(x)
        return x
    
    
# ###################################################
# ### ////// Conditional Generator Network ////// ###
# ###################################################

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

class ConditiningNetwork(torch.nn.Module):
    def __init__(self, sequence_lenght=196, feature_dim=192, action_code_dim=12, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1):
        super(ConditiningNetwork, self).__init__()

        self.sequence_length = sequence_lenght + 1  # 196 feature tokens + 1 action token
        self.feature_dim = feature_dim
        self.action_code_dim = action_code_dim
        
        # MLP to transform the action code into a 192-D token
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_code_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim))
        # MLP to map features tokens into a space of the same dimension. This helps by having feature tokens in the same space as the action code token
        self.feature_mlp = nn.Linear(self.feature_dim, self.feature_dim)
        # Positional Encoding for the sequence
        self.positional_encoding = PositionalEncoding(d_model=self.feature_dim, max_len=self.sequence_length)
        # Transformer network
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   layer_norm_eps=1e-6, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feature_tokens, action_code):
        """
        action_code: Tensor of shape (batch_size, 12)
        feature_tokens: Tensor of shape (batch_size, 196, 192)
        Returns:
            transformed_feature_tokens: Tensor of shape (batch_size, 196, 192)
        """
        # Step 1: Transform action code into a 192-D token
        action_token = self.action_mlp(action_code)  # shape: (batch_size, 192)
        action_token = action_token.unsqueeze(1)  # shape: (batch_size, 1, 192)
        # Step 2: Apply a linear layer feature tokens
        feature_tokens = self.feature_mlp(feature_tokens) # shape: (batch_size, 196, 192)
        # Step 3: Concatenate the action token with feature tokens (action token first)
        tokens = torch.cat((action_token, feature_tokens), dim=1)  # shape: (batch_size, 197, 192)
        # Step 4: Add positional encoding
        tokens = self.positional_encoding(tokens) # shape: (batch_size, 197, 192)
        # Step 5: Pass through the Transformer Encoder (it has batch_first=True, so we are good)
        transformed_tokens = self.transformer_encoder(tokens)  # shape: (batch_size, 197, 192)
        # Step 7: Drop the first token (action code) and reshape
        transformed_feature_tokens = transformed_tokens[:, 1:, :]  # shape: (batch_size, 196, 192)

        return transformed_feature_tokens
    
# class DecoderNetwork_transformer(torch.nn.Module):
#     def __init__(self, sequence_lenght=196, feature_dim=192, num_layers=4, nhead=6, dim_feedforward=512, dropout=0.1):
    



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
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate')
        self.norm2 = nn.GroupNorm(min([32, in_planes//4]), in_planes)
        self.act = nn.Mish()
        if stride == 1: # Not changing spatial size
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate')
            self.norm1 = nn.GroupNorm(min([32, planes//4]), planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate'),
                nn.GroupNorm(min([32, planes//4]), planes)
            )
        else: # Changing spatial size (expanded by x2)
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='replicate')
            self.norm1 = nn.GroupNorm(min([32, planes//4]), planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='replicate'),
                nn.GroupNorm(min([32, planes//4]), planes)
            )
    def forward(self, x):
        out = self.act(self.norm2(self.conv2(x)))
        out = self.norm1(self.conv1(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class DecoderNetwork_convolution(nn.Module):
    def __init__(self,
                 in_planes=192,
                 num_Blocks=[1,1,1,1], 
                 nc=3):
        super().__init__()
        self.in_planes = in_planes
        self.out_act = nn.Tanh() # Because we are reconstructing input images with values between -1 and 1

        # Since the feature tokens start already at 14x14, we make layer 4 to output the same spatial size by doing stride 1.
        # (This is the case for ViT with 196 tokens (patch size of 16 on 224x224 images)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=1) # stride 1 to not change spatial size
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)

        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='replicate') ## 3x3 kernel size

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
        # x shape input is (batch_size, 196, 192). Let's reshape it to (batch_size, 192, 14, 14)
        x = x.permute(0, 2, 1) # (batch_size, 192, 196)
        h = math.sqrt(x.shape[-1]) # 14
        w = math.sqrt(x.shape[-1]) # 14
        x = x.reshape(x.shape[0], x.shape[1], int(h), int(w)) # (batch_size, 192, 14, 14)
        # Now we can pass it through the decoder
        x = self.layer4(x) # (batch_size, 256, 14, 14)
        x = self.layer3(x) # (batch_size, 128, 28, 28)
        x = self.layer2(x) # (batch_size, 64, 56, 56)
        x = self.layer1(x) # (batch_size, 64, 112, 112)
        x = self.out_act(self.conv1(x)) # (batch_size, 3, 224, 224)
        return x

class ConditionalGenerator(nn.Module):
    def __init__(self,
                 action_code_dim = 12,
                 feature_dim = 192,
                 ft_num_layers = 2, 
                 ft_nhead = 4, 
                 ft_dim_feedforward = 256, 
                 ft_dropout = 0.1,
                 dec_num_Blocks = [1,1,1,1],
                 dec_num_out_channels = 3,
                 sequence_lenght = 196, # Number of feature tokens
                 ):
        super(ConditionalGenerator, self).__init__()

        # Define conditioning network
        self.conditioning_network = ConditiningNetwork(sequence_lenght=sequence_lenght,
                                                        feature_dim=feature_dim, 
                                                        action_code_dim=action_code_dim, 
                                                        num_layers=ft_num_layers, 
                                                        nhead=ft_nhead, 
                                                        dim_feedforward=ft_dim_feedforward, 
                                                        dropout=ft_dropout)
        # Define decoder
        self.decoder = DecoderNetwork_convolution(in_planes=feature_dim , num_Blocks=dec_num_Blocks, nc=dec_num_out_channels)

    def forward(self, feature_map, action_code, skip_conditioning=False):
        # Feature map shape: (batch_size, 196, 192)
        # Action code shape: (batch_size, 12)
        if skip_conditioning: 
            # Pass tensor direcly to the decoder to boost learning (It helps training the generator to create better images)
            generated_image = self.decoder(feature_map)
            return generated_image
        else: 
            # Use the feature transformation network to learn to predict the next view tensor using action code and previous view.
            # This option also works when action code means "no action" (i.e., the next view is the same as the previous view)
            transformed_feature_map = self.conditioning_network(feature_map, action_code)
            generated_image = self.decoder(transformed_feature_map)
            return generated_image, transformed_feature_map

    
### Test models in main
if __name__ == '__main__':
    import torch
    from torchinfo import summary
    batch_size=2

    # Test encoder
    model = deit_tiny_patch16_LS(img_size=224, num_classes=1000)
    print(model)
    x = torch.randn(batch_size, 3, 224, 224)
    y = model(x)
    print('Output shape:', y.shape)
    summary(model, input_size=(batch_size, 3, 224, 224), device='cpu')

    # Test classifier
    model = Classifier_Model(input_dim=768, num_classes=1000)
    print(model)
    x = torch.randn(batch_size, 768)
    y = model(x)
    print('Output shape:', y.shape)
    summary(model, input_size=(batch_size, 768), device='cpu')

    # Test ConditiningNetwork
    model = ConditiningNetwork(sequence_lenght=196, feature_dim=192, action_code_dim=12)
    print(model)
    x = torch.randn(batch_size, 196, 192)
    action_code = torch.randn(batch_size, 12)
    y = model(x, action_code)
    print('Output shape:', y.shape)

    # Test DecoderNetwork_convolution
    model = DecoderNetwork_convolution(in_planes=192, num_Blocks=[1,1,1,1], nc=3)
    print(model)
    x = torch.randn(batch_size, 196, 192)
    y = model(x)
    print('Output shape:', y.shape)
    summary(model, input_size=(batch_size, 196, 192), device='cpu')

    # Test ConditionalGenerator
    model = ConditionalGenerator(action_code_dim=12, feature_dim=192, ft_num_layers=2, ft_nhead=4, ft_dim_feedforward=256, ft_dropout=0.1, dec_num_Blocks=[1,1,1,1], dec_num_out_channels=3)
    print(model)
    x = torch.randn(batch_size, 196, 192)
    action_code = torch.randn(batch_size, 12)
    y = model(x, action_code)
    print('Output Generate Image shape:', y[0].shape)
    print('Output Transformed Feature Map shape:', y[1].shape)
    y = model(x, action_code, skip_conditioning=True)
    print('Output Generate Image shape:', y.shape)


