# models/DSTformer3.py

import torch
import torch.nn as nn
from collections import OrderedDict
from models.drop import DropPath

def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # [Batch, Sequence Length, Embedding Dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is [Batch, Heads, Seq Length, Head Dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [Batch, Heads, Seq Length, Seq Length]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [Batch, Seq Length, Embedding Dim]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # x: [Seq Length, Batch, dim]
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm)
        x = x + self.drop_path(attn_output)
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + self.drop_path(mlp_output)
        return x

class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, dim_rep=256,
                 depth=5, num_heads=8, num_joints=20, maxlen=50,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super(DSTformer, self).__init__()
        self.dim_feat = dim_feat
        self.dim_rep = dim_rep
        self.num_joints = num_joints
        self.maxlen = maxlen
        print(f"Initializing DSTformer with maxlen={maxlen}")

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Learnable [CLS] tokens
        self.cls_token_spatial = nn.Parameter(torch.zeros(1, 1, dim_feat))
        self.cls_token_temporal = nn.Parameter(torch.zeros(1, 1, dim_feat))
        trunc_normal_(self.cls_token_spatial, std=.02)
        trunc_normal_(self.cls_token_temporal, std=.02)

        # Positional Embeddings
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 1, dim_feat))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, maxlen + 1, dim_feat))
        trunc_normal_(self.spatial_pos_embed, std=.02)
        trunc_normal_(self.temporal_pos_embed, std=.02)

        self.out_size = dim_rep

        self.blocks_spatial = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.blocks_temporal = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        self.norm_spatial = norm_layer(dim_feat)
        self.norm_temporal = norm_layer(dim_feat)

        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        print(f"x device: {x.device}")
        print(f"self.joints_embed.weight device: {self.joints_embed.weight.device}")
        # x: [B, C, T, J]
        print(f"Initial x shape: {x.shape}")
        B, C, T, J = x.shape

        # Permute to [B, T, J, C]
        x = x.permute(0, 2, 3, 1)
        print(f"x after permute: {x.shape}")

        B, T, J, C = x.shape
        x = x.reshape(B * T, J, C)  # [B*T, J, C]

        # Apply joints_embed per joint
        x = self.joints_embed(x)  # [B*T, J, dim_feat]
        print(f"x after joints_embed: {x.shape}")

        x = x.view(B, T, J, self.dim_feat)  # [B, T, J, dim_feat]
        print(f"x reshaped to [B, T, J, dim_feat]: {x.shape}")

        x = self.pos_drop(x)

        # Spatial Processing
        cls_tokens_spatial = self.cls_token_spatial.expand(B, T, 1, -1)  # [B, T, 1, dim_feat]
        x = torch.cat((cls_tokens_spatial, x), dim=2)  # [B, T, J + 1, dim_feat]
        x = x + self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)  # Broadcast over B and T

        x = x.view(B * T, self.num_joints + 1, self.dim_feat).permute(1, 0, 2)  # [J + 1, B*T, dim_feat]
        for blk in self.blocks_spatial:
            x = blk(x)
        x = x.permute(1, 0, 2).contiguous()  # [B*T, J + 1, dim_feat]
        x = self.norm_spatial(x)
        spatial_tokens = x.view(B, T, self.num_joints + 1, self.dim_feat)
        print(f"spatial_tokens shape: {spatial_tokens.shape}")

        # Extract [CLS] token for spatial global features
        cls_spatial_output = spatial_tokens[:, :, 0, :]  # [B, T, dim_feat]
        spatial_global = self.pre_logits(cls_spatial_output.mean(dim=1))  # [B, dim_rep]

        # Temporal Processing
        x = spatial_tokens[:, :, 1:, :]  # Exclude [CLS] token; [B, T, J, dim_feat]
        x = x.permute(0, 2, 1, 3)  # [B, J, T, dim_feat]
        x = x.reshape(B * J, T, self.dim_feat)  # [B*J, T, dim_feat]

        cls_tokens_temporal = self.cls_token_temporal.expand(B * J, 1, -1)  # [B*J, 1, dim_feat]
        x = torch.cat((cls_tokens_temporal, x), dim=1)  # [B*J, T + 1, dim_feat]
        x = x + self.temporal_pos_embed[:, :T + 1, :].repeat(B * J, 1, 1)
        x = x.permute(1, 0, 2)  # [T + 1, B*J, dim_feat]

        for blk in self.blocks_temporal:
            x = blk(x)
        x = x.permute(1, 0, 2).contiguous()  # [B*J, T + 1, dim_feat]
        x = self.norm_temporal(x)

        temporal_tokens = x.view(B, J, T + 1, self.dim_feat)
        print(f"temporal_tokens shape: {temporal_tokens.shape}")

        # Extract [CLS] token for temporal global features
        cls_temporal_output = temporal_tokens[:, :, 0, :]  # [B, J, dim_feat]
        temporal_global = self.pre_logits(cls_temporal_output.mean(dim=1))  # [B, dim_rep]

        global_features = spatial_global + temporal_global  # [B, dim_rep]

        local_features = {
            'spatial_tokens': spatial_tokens,      # [B, T, J + 1, dim_feat]
            'temporal_tokens': temporal_tokens     # [B, J, T + 1, dim_feat]
        }

        return global_features, local_features
