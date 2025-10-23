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
    'Classifier_Network',
    'Action_Encoder_Network',
    'View_Predictor_Network',
    'Generator_Network'
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
        self.num_reg_tokens = 16

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, embed_dim))

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
        trunc_normal_(self.register_tokens, std=.02)
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
        return {'pos_embed', 'cls_token', 'register_tokens'}

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
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, register_tokens), dim=1)
        
        # Apply blocks
        for i , blk in enumerate(self.blocks):
            x = blk(x)
        # Normalize final output (if pooling is avg, do normalization after pooling)
        x = self.norm(x)

        # Remove register tokens
        x = x[:, :-self.num_reg_tokens]

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


class Classifier_Network(torch.nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()

        ### Projector
        hidden_dim = 1024
        bottleneck_dim = 256
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), #expand
            torch.nn.LayerNorm(hidden_dim, eps=1e-6),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck_dim, bias=False), # bottleneck
        )

        #### Classifier_head
        self.norm = L2Norm(dim=1) # L2 normalization
        self.tau = 0.1 # temperature for cosine softmax
        self.classifier_head = torch.nn.Linear(bottleneck_dim, num_classes, bias=False) 

    def forward(self, x):
        # shape of x is (B*V, D)
        # Projector
        x = self.projector(x) # (B*V, bottleneck_dim)
        # Cosine similarity classifier
        x = self.norm(x) # L2 normalization of features
        w = self.norm(self.classifier_head.weight) # L2 normalization of classifier weights
        o = (x @ w.t()) / self.tau # (B*V, num_classes)
        return o

######################################################
### ////// Action Encoder Generator Network ////// ###
######################################################

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
    
    def _tok(self, name, params):
        # Concatenate type embedding and parameters
        dev = self.type_emb.weight.device 
        idx = torch.tensor([self.name2id[name]], device=dev)
        t = self.type_emb(idx)                       # (1,D)
        head = self.proj[name]
        if head is not None and params.numel():
            t = torch.cat([t, head(params.to(dev).unsqueeze(0))], dim=1) # (1,D)
        else:
            t = torch.cat([t, torch.zeros(1, self.d_linparam, device=dev)], dim=1) # (1,D)
        return t

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

class Action_Encoder_Network(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4, dim_ff=256):
        super().__init__()
        self.dim_type_emb = d_model // 2
        self.dim_linparam = d_model // 2

        # Action Tokenizer
        self.aug_tokeniser = AugTokenizerSparse(d_type_emb=self.dim_type_emb, d_linparam=self.dim_linparam)

        # Positional encoding
        self.pe_aug = SinCosPE(self.dim_type_emb, 16)
        
        # Transformer encoder
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

    def forward(self, actions):
        # Actions are (B*V, A)
        N = len(actions) # N = B*V
        # Tokenize actions
        aug_tokens, pad_masks = self.aug_tokeniser(actions)    # (B*V, Taug, Daug),(B*V, Taug), Taug is the maximum number of augmentation operations for any action in the batch
        # Add positional encoding
        aug_tokens = aug_tokens + self.pe_aug(aug_tokens.size(1), append_zeros_dim=self.dim_linparam)
        # Encode all L tokens
        h = self.enc(aug_tokens, src_key_padding_mask=pad_masks)   # (B*V, L, D)
        # Expand k queries to batch
        q = self.pool_q.unsqueeze(0).expand(N, -1, -1)   # (B*V, k, D)
        # Attend k queries over the L outputs
        summaries, _ = self.pool_attn(
            q, h, h, key_padding_mask=pad_masks            # (B*V, k, D)
        )
        # Collapse k → 1 by mean
        summary = summaries.mean(dim=1, keepdim=True)    # (B*V, 1, D)
        return summary


############################################
### ////// View Predictor Network ////// ###
############################################

class View_Predictor_Network(nn.Module):
    def __init__(self, 
                 d_model=256,
                 n_img_tokens=196, 
                 imgfttok_dim=192,
                 acttok_dim=64, 
                 num_layers=8, 
                 nhead=8, 
                 dim_ff=1024, 
                 dropout=0):
        super(View_Predictor_Network, self).__init__()
        
        ### Dimension of token input to transformer. We project all tokens to the same dimension.
        self.hidden_dim = d_model

        ### Input projections per token type
        self.imgfttok_mlp_in  = nn.Linear(imgfttok_dim, self.hidden_dim)
        self.acttok_mlp_in    = nn.Linear(acttok_dim, self.hidden_dim)
        self.norm_in          = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        ### Output projections per token type
        self.imgfttok_mlp_out = nn.Sequential(nn.Linear(self.hidden_dim, imgfttok_dim),
                                              nn.LayerNorm(imgfttok_dim, eps=1e-6))

        ### Type embeddings per token type
        self.type_emb_imgfttok = nn.Parameter(torch.zeros(self.hidden_dim))
        self.type_emb_acttok = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.type_emb_imgfttok, std=0.02)
        trunc_normal_(self.type_emb_acttok, std=0.02)

        ### pos-encodings
        self.pe = SinCosPE(self.hidden_dim, 5000) 

        ### Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            layer_norm_eps=1e-6, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        ### Mask token
        self.mask_imgtoken = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.mask_imgtoken, std=0.02)

        ### Type embeddings for mask tokens
        self.type_emb_maskimgtok = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.type_emb_maskimgtok, std=0.02)


    def forward(self, noflat_imgfttoks, noflat_acttok):
        """
        Inputs:
          noflat_imgfttoks: (B, V, Timg, Dimg)
          noflat_acttok:    (B, V, 1,    Dact)
        Returns:
          noflat_PRED_imgfttoks: (B, V, Timg, Dimg)  predictions for each view's image tokensn
        """

        B, V, Timg, Dimg = noflat_imgfttoks.shape
        _, _, _, Dacttok = noflat_acttok.shape
        Dhidden = self.hidden_dim

        noflat_PRED_imgfttoks = torch.zeros_like(noflat_imgfttoks, device=noflat_imgfttoks.device) # (B, V, Timg, Dimg)

        # 1) Project the input tokens to the hidden dimension and normalize
        noflat_imgfttoks_hidden = self.norm_in(self.imgfttok_mlp_in(noflat_imgfttoks)) # (B, V, Timg, Dhidden)
        noflat_acttok_hidden = self.norm_in(self.acttok_mlp_in(noflat_acttok)) # (B, V, 1, Dhidden)

        # 2) Add type embeddings
        noflat_imgfttoks_hidden = noflat_imgfttoks_hidden + self.type_emb_imgfttok.reshape(1,1,1,Dhidden).expand(B,V,Timg,Dhidden) # (B, V, Timg, Dhidden)
        noflat_acttok_hidden = noflat_acttok_hidden + self.type_emb_acttok.reshape(1,1,1,Dhidden).expand(B,V,1,Dhidden) # (B, V, 1, Dhidden)

        # 3) Concatenate the tokens for each view and reshape to (B, V*(1+Timg+1), Dhidden)
        base_seqs = torch.cat((noflat_acttok_hidden, noflat_imgfttoks_hidden), dim=2) # (B, V, 1+Timg, Dhidden)
        base_seqs = base_seqs.reshape(B, V*(1+Timg), Dhidden) # (B, V*(1+Timg), Dhidden)

        # 4) Pre-compute positional encoding
        pe = self.pe(base_seqs.size(1)) # (1, V*(1+Timg), Dhidden)

        # 5) Normalize mask token and add type embedding
        mask_imgtoken = self.norm_in(self.mask_imgtoken) + self.type_emb_maskimgtok # (Dhidden)

        # 7) Generate a random mask (like in MAE). We won't replace the complete view with mask tokens, only a subset of the tokens selected by the random mask.
        ratio=0.75 #0.95 # 95% of the tokens of the current view will be masked
        num_masked_tokens = int(Timg * ratio)
        mask_indices = torch.randperm(Timg)[:num_masked_tokens]

        # 7) Predict current view by replacing its input tokens with mask tokens (dev3)-> Include view 1
        for i in range(V):
            # Clone sequences
            seqs = base_seqs.clone() # (B, V*(1+Timg), Dhidden)
            # Define start and end of mask
            start = i*(1+Timg)+1
            end = (i+1)*(1+Timg)
            # Mask some of the tokens of the current view
            seqs[:, start:end, :][:, mask_indices, :] = mask_imgtoken.reshape(1, 1, Dhidden).expand(B, num_masked_tokens, Dhidden)
            # Add positional encoding
            seqs = seqs + pe[:, :seqs.size(1), :]
            # Encode
            seqs_out = self.transformer_encoder(seqs) # (B, V*(1+Timg), Dhidden)
            # Collect predictions at the masked positions
            pred_img_hidden = seqs_out[:, start:end, :] # (B, Timg, Dhidden)
            noflat_PRED_imgfttoks[:, i, :, :] = self.imgfttok_mlp_out(pred_img_hidden) # (B, Timg, Dimg)

        return noflat_PRED_imgfttoks, mask_indices

#######################################
### ////// Generator Network ////// ###
#######################################

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

class Generator_Network(nn.Module):
    def __init__(self,
                 in_planes=192,
                 num_Blocks=[1,1,1,1], 
                 nc=3):
        super().__init__()
        self.in_planes = in_planes
        self.out_act = lambda x: torch.tanh(x)

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
        # Input x is shape (B, V, Timg, Dimg)
        B, V, Timg, Dimg = x.shape
        # Reshape x to (B*V, Timg, Dimg)
        x = x.reshape(B*V, Timg, Dimg)
        # Reshape it to (B*V, Dimg, 14, 14)
        x = x.permute(0, 2, 1) # (B*V, Dimg, Timg)
        h = math.sqrt(Timg) # 14 if Timg = 196
        w = math.sqrt(Timg) # 14 if Timg = 196
        x = x.reshape(x.shape[0], x.shape[1], int(h), int(w)) # (B*V, Dimg, h, w)
        # Now we can pass it through the decoder
        x = self.layer4(x) # (B*V, 256, h, w)
        x = self.layer3(x) # (B*V, 128, 28, 28)
        x = self.layer2(x) # (B*V, 64, 56, 56)
        x = self.layer1(x) # (B*V, 64, 112, 112)
        x = self.out_act(self.conv1(x)) # (B*V, 3, 224, 224)
        # Reshape x to (B, V, 3, 224, 224)
        x = x.reshape(B, V, 3, 224, 224)
        return x

    





