# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.layers import DropPath, to_2tuple, trunc_normal_

from torch.nn.utils.rnn import pad_sequence

__all__ = [
    'deit_tiny_patch16_LS',
    'deit_small_patch16_LS',
    'deit_medium_patch16_LS',
    'deit_base_patch16_LS',
    'deit_large_patch16_LS',
    'Classifier_Model',
    'ConditionalGenerator'
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
        
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = F.scaled_dot_product_attention(q, k, v) # Flash Attention
        x = x.transpose(1,2).reshape(B, N, C)

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

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return "dim={}".format(self.dim)

    def forward(self, x):
        return nn.functional.normalize(x, dim=self.dim, p=2)

class CapturableEncoderLayer(nn.TransformerEncoderLayer):
    """
    Same as nn.TransformerEncoderLayer, but stores self-attention probabilities
    used in the forward pass in `self.last_attn_probs` (shape (B, H, T, S)).
    Works with batch_first=True. Keeps norm_first behavior from parent.
    """
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal: bool = False):
        # Ask MHA for weights (probabilities after softmax)
        attn_out, attn_probs = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head (H)
            is_causal=is_causal,
        )
        # attn_probs comes in one of two shapes depending on PyTorch version:
        #  - (B, H, T, S)  (preferred, when batch_first=True)
        #  - (B*H, T, S)   (older path); reshape it
        if attn_probs.dim() == 3:
            B = x.size(0)        # batch
            T = x.size(1)        # query length
            S = attn_probs.size(-1)
            H = self.self_attn.num_heads
            attn_probs = attn_probs.reshape(B, H, T, S)

        self.last_attn_probs = attn_probs  # keep for loss; requires_grad=True
        return self.dropout1(attn_out)

class Classifier_Model(torch.nn.Module):
    def __init__(self, input_dim, n_heads=1, n_layers=1, dropout=0.1, num_classes=1000):
        super().__init__()

        ### Projector
        hidden_dim = 1024
        bottleneck_dim = 256
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), #expand
            torch.nn.LayerNorm(hidden_dim, eps=1e-6),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck_dim, bias=False), # bottleneck
            torch.nn.LayerNorm(bottleneck_dim, eps=1e-6),
        )

        ### Positional encoding
        self.pos_embed = SinCosPE(bottleneck_dim, max_length=20) # a maximum of 20 views

        ### Causal transformer head
        transf_layer = CapturableEncoderLayer(
            bottleneck_dim, n_heads, dim_feedforward=bottleneck_dim*4,
            batch_first=True, activation='gelu',
            layer_norm_eps=1e-6, dropout=dropout
        )
        self.transf = nn.TransformerEncoder(transf_layer, n_layers)


        #### Classifier_head
        self.norm = L2Norm(dim=1) # L2 normalization
        self.tau = 0.1 # temperature for cosine softmax
        self.classifier_head = torch.nn.Linear(bottleneck_dim, num_classes, bias=False) 


    def forward(self, x):
        # shape of x is (B, T, D) where B is batch size, T is number of tokens, D is feature dimension
        B, T, D = x.shape
        # Reshape B, T, D to B*T, D
        x = x.reshape(B * T, D) # (B*T, D)
        # Projector
        x = self.projector(x) # (B*T, bottleneck_dim)
        # Reshape back to (B, T, bottleneck_dim)
        x = x.reshape(B, T, -1) # (B, T, bottleneck_dim)
        # Add positional encoding
        x = x + self.pos_embed(T) # (B, T, bottleneck_dim)
        # causal transformer
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(x.device) # (T,T)
        x = self.transf(x, mask=causal_mask, is_causal=True) # (B, T, bottleneck_dim)
        # Reshape B, T, D to B*T, D
        x = x.reshape(B * T, -1) # (B*T, bottleneck_dim)

        # Cosine similarity classifier
        x = self.norm(x) # L2 normalization of features
        w = self.norm(self.classifier_head.weight) # L2 normalization of classifier weights
        o = (x @ w.t()) / self.tau # (B*T, num_classes)
        o = o.reshape(B, T, -1)
        return o
    
    
# ###################################################
# ### ////// Conditional Generator Network ////// ###
# ###################################################

class SinCosPE(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        pe = torch.zeros(max_length, dim)
        pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0)/dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, seq_length, append_zeros_dim=None):
        if append_zeros_dim is None:
            return self.pe[: seq_length].unsqueeze(0)
        else:
            pe = self.pe[: seq_length]
            zeros = torch.zeros(pe.size(0), append_zeros_dim, device=pe.device)
            return torch.cat([pe, zeros], dim=1).unsqueeze(0)

class AugTokenizerSparse(nn.Module):
    def __init__(self, d_type_emb=32, d_linparam=32):
        super().__init__()
        self.d_type_emb = d_type_emb
        self.d_linparam = d_linparam
        self.d = d_type_emb + d_linparam
        self.name2id = {
            "crop": 0, "hflip": 1, "jitter": 2,
            "gray": 3, "blur": 4, "solar": 5,
            "none": 6,                      # emitted when list is empty
        }
        self.type_emb = nn.Embedding(len(self.name2id), d_type_emb)

        self.proj = nn.ModuleDict({
            "crop"  : nn.Linear(4, d_linparam), # process 4 params
            "hflip" : None,                  # no params to process
            "jitter": nn.Linear(7, d_linparam), # process 7 params
            "gray"  : None,                  # no params to process
            "blur"  : nn.Linear(1, d_linparam), # process 1 param
            "solar" : nn.Linear(1, d_linparam), # process 1 param
            "none"  : None,                  # no params to process
        })

        self.pad_emb  = nn.Parameter(torch.zeros(1, self.d))   # <PAD>

    # create a single token --------------------------------------------------
    def _tok(self, name, params):
        dev = self.type_emb.weight.device 
        idx = torch.tensor([self.name2id[name]], device=dev)
        t = self.type_emb(idx)                       # (1,D)
        head = self.proj[name]
        if head is not None and params.numel():
            t = torch.cat([t, head(params.to(dev).unsqueeze(0))], dim=1) # (1,D)
        else:
            t = torch.cat([t, torch.zeros(1, self.d_linparam, device=dev)], dim=1) # (1,D)
        return t

    # main forward -----------------------------------------------------------
    def forward(self, batch_aug_lists):
        """
        batch_aug_lists = List[List[(name:str, params:Tensor)]], len = B
        returns padded_tokens (B,Lmax,2*D) , pad_mask (B,Lmax)
        """

        device = self.type_emb.weight.device
        used   = {k: False for k in self.proj}          # track usage
        seqs = []
        for ops in batch_aug_lists:
            if len(ops) == 0:
                seqs.append(self._tok("none", torch.empty(0, device=device)))
                used["none"] = True
            else:
                toks = []
                for name, p in ops:
                    toks.append(self._tok(name, p))
                    used[name] = True
                seqs.append(torch.cat(toks, dim=0))

        Lmax   = max(s.size(0) for s in seqs)
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.)
        lengths= torch.tensor([s.size(0) for s in seqs], device=padded.device)
        mask   = torch.arange(Lmax, device=padded.device)[None, :] >= lengths[:, None]

        padded[mask] = self.pad_emb                # replace zeros with <PAD>

        # ---- one dummy call per *unused* head ----------------------------
        # This is so DDP doesn't complain about unused heads. 
        # Using find_unused_parameters=True didn't help because I call the network twice (upsampling resnet)
        # one during generated FTN feature image generation, another one with the "direct" use of encoder features for image generation.
        # That causes the find_unused_parameters to complain for doing double marking (marking used twice).
        # I found this solution here to work which is just calling the unused heads with a dummy input * 0 so it doesn't affect the output.
        dummy_sum = 0.0
        dummy_type = torch.zeros(1, self.d_type_emb, device=self.type_emb.weight.device)
        for name, flag in used.items():
            head = self.proj[name]
            if head is not None and not flag:          # unused in this batch
                z = torch.zeros(head.in_features, device=device)
                dummy_sum = dummy_sum + torch.cat([dummy_type, head(z.unsqueeze(0))], dim=1) # (1,D)
        padded = padded + 0.0 * dummy_sum              # attach, keep value

        return padded, mask                        # (B,L,D), (B,L)

class AugEncoder(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4, dim_ff=256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=dim_ff,
            batch_first=True, activation='gelu',
            layer_norm_eps=1e-6
        )
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)

        # k learnable queries to pool L -> k
        num_queries = 4
        self.num_q     = num_queries
        self.pool_q    = nn.Parameter(torch.zeros(num_queries, d_model))  # (k, D)
        trunc_normal_(self.pool_q, std=0.02)

        # multihead-attn to pool: Q=pool_q, K=enc_out, V=enc_out
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )

    def forward(self, x, pad_mask):             # (B,L,D),(B,L)
        # 1) encode all L tokens
        h = self.enc(x, src_key_padding_mask=pad_mask)   # (B, L, D)

        # 2) expand k queries to batch
        B, L, D = h.shape
        q = self.pool_q.unsqueeze(0).expand(B, -1, -1)   # (B, k, D)

        # 3) pool: attend k queries over the L outputs
        summaries, _ = self.pool_attn(
            q, h, h, key_padding_mask=pad_mask            # (B, k, D)
        )

        # 4) collapse k → 1 by mean
        summary = summaries.mean(dim=1, keepdim=True)    # (B, 1, D)
        return summary
    
class ConditioningNetwork(nn.Module):
    def __init__(self, 
                 n_img_tokens=196, 
                 feature_dim=192, 
                 num_layers=8, 
                 nhead=8, 
                 dim_ff=1024, 
                 dropout=0, 
                 max_aug_tokens=16,
                 aug_feature_dim=64,
                 aug_n_layers=2,
                 aug_n_heads=4,
                 aug_dim_ff=256):
        super(ConditioningNetwork, self).__init__()
        self.dim_type_emb = aug_feature_dim // 2
        self.dim_linparam = aug_feature_dim // 2

        # pos-encodings
        self.pe_img = SinCosPE(feature_dim, n_img_tokens)
        self.pe_aug = SinCosPE(self.dim_type_emb, max_aug_tokens) # only add pos to the first half of the token (type embedding section)

        # augmentation side
        self.aug_tokeniser = AugTokenizerSparse(d_type_emb=self.dim_type_emb, d_linparam=self.dim_linparam)
        self.aug_enc = AugEncoder(d_model=aug_feature_dim, n_layers=aug_n_layers, 
                                    n_heads=aug_n_heads, dim_ff=aug_dim_ff)
        self.aug_mlp = nn.Sequential(
                                nn.Linear(aug_feature_dim, feature_dim),
                                nn.LayerNorm(feature_dim, eps=1e-6)
        )
        
        # linear to re-embed image tokens in same space
        self.feature_mlp = nn.Sequential(
                                nn.Linear(feature_dim, feature_dim),
                                nn.LayerNorm(feature_dim, eps=1e-6)
        )

        # Transformer decoder (cross-attn = image queries  ←→ aug keys/values)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            layer_norm_eps=1e-6, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

    def forward(self, feature_tokens_input, aug_seq_batch):
        # 1) build & encode augmentation tokens
        aug_tok, pad_mask = self.aug_tokeniser(aug_seq_batch)    # (B,L,D),(B,L)
        aug_tok = aug_tok + self.pe_aug(aug_tok.size(1), append_zeros_dim=self.dim_linparam)     # (B,L,D)

        # Encode the augmentation tokens
        summary  = self.aug_enc(aug_tok, pad_mask)            # (B,1,D)
        memory  = self.aug_mlp(summary)                       # (B,1,d)

        # 2) embed + PE for image tokens (= decoder queries)
        tgt = self.feature_mlp(feature_tokens_input)

        # 3) add PE for image tokens
        tgt = tgt + self.pe_img(tgt.size(1))                              # (B,196,d)

        # 4) cross-attention
        out = self.transformer_decoder(tgt=tgt,
                                    memory=memory) # (B,196,d)
        return out

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, padding=1, padding_mode='reflect', mode='bilinear', bias=False): # bilinear
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias, padding_mode=padding_mode)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, antialias=True)
        x = self.conv(x)
        return x
    
class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(min([32, in_planes//4]), in_planes)
        self.act = nn.Mish()
        if stride == 1: # Not changing spatial size
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
            self.norm1 = nn.GroupNorm(min([32, planes//4]), planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
                nn.GroupNorm(min([32, planes//4]), planes)
            )
        else: # Changing spatial size (expanded by x2)
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='reflect')
            self.norm1 = nn.GroupNorm(min([32, planes//4]), planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='reflect'),
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
        self.out_act = lambda x: 3.0 * torch.tanh(x) # My tanh so it can predict values up to 2.5 (input image statistics have those values)

        # Since the feature tokens start already at 14x14, we make layer 4 to output the same spatial size by doing stride 1.
        # (This is the case for ViT with 196 tokens (patch size of 16 on 224x224 images)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=1) # stride 1 to not change spatial size
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)

        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='reflect', bias=True) ## 3x3 kernel size
        # init conv1 bias as zero
        nn.init.constant_(self.conv1.conv.bias, 0)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Similar to zero init residual
        for m in self.modules():
            if isinstance(m, BasicBlockDec) and m.norm1.weight is not None:
                nn.init.constant_(m.norm1.weight, 0) # shutdown the main path and only let the residual path pass at init

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
                 img_num_tokens=196,
                 img_feature_dim = 192,
                 num_layers = 2, 
                 nhead = 4, 
                 dim_ff = 256, 
                 dropout = 0.1,
                 aug_num_tokens_max = 16,
                 aug_feature_dim = 24,
                 aug_n_layers = 2,
                 aug_n_heads = 4,
                 aug_dim_ff = 128,
                 upsampling_num_Blocks = [1,1,1,1],
                 upsampling_num_out_channels = 3,
                 ):
        super(ConditionalGenerator, self).__init__()

        # Define conditioning network
        self.conditioning_network = ConditioningNetwork(n_img_tokens=img_num_tokens,
                                                        feature_dim=img_feature_dim, 
                                                        num_layers=num_layers, 
                                                        nhead=nhead, 
                                                        dim_ff=dim_ff, 
                                                        dropout=dropout,
                                                        max_aug_tokens=aug_num_tokens_max,
                                                        aug_feature_dim=aug_feature_dim,
                                                        aug_n_layers=aug_n_layers,
                                                        aug_n_heads=aug_n_heads,
                                                        aug_dim_ff=aug_dim_ff)
        # Define decoder
        self.decoder = DecoderNetwork_convolution(in_planes=img_feature_dim , num_Blocks=upsampling_num_Blocks, nc=upsampling_num_out_channels)

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





