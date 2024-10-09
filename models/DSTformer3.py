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
                 attn_drop=0., proj_drop=0., attention_type='spatial'):
        super().__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, attention_type='spatial'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop,
                              attention_type=attention_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, dim_rep=256,
                 depth=5, num_heads=8, num_joints=20, maxlen=50,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        """
        DSTformer for processing skeleton data.

        Args:
            dim_in (int): Input feature dimension per joint.
            dim_feat (int): Feature dimension after embedding.
            dim_rep (int): Representation dimension after pre-logits.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            num_joints (int): Number of joints in the skeleton.
            maxlen (int): Maximum sequence length.
            mlp_ratio (float): MLP ratio in transformer blocks.
            drop_rate (float): Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Drop path rate.
            norm_layer (nn.Module): Normalization layer.
        """
        # print(f"Initializing DSTformer with dim_in={dim_in}, dim_feat={dim_feat}, "
        #       f"dim_rep={dim_rep}, depth={depth}, num_heads={num_heads}, "
        #       f"num_joints={num_joints}, maxlen={maxlen}, mlp_ratio={mlp_ratio}, "
        #       f"drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, "
        #       f"drop_path_rate={drop_path_rate}")
        
        super().__init__()
        self.dim_feat = dim_feat
        self.dim_rep = dim_rep
        self.num_joints = num_joints
        self.maxlen = maxlen

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, maxlen, dim_feat))
        trunc_normal_(self.spatial_pos_embed, std=.02)
        trunc_normal_(self.temporal_pos_embed, std=.02)

        self.blocks_spatial = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate, norm_layer=norm_layer,
                  attention_type='spatial')
            for _ in range(depth)
        ])
        self.blocks_temporal = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate, norm_layer=norm_layer,
                  attention_type='temporal')
            for _ in range(depth)
        ])
        self.norm = norm_layer(dim_feat)
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
        """
        Forward pass for DSTformer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, J, C=3]

        Returns:
            spatial_tokens (torch.Tensor): Spatial tokens of shape [B, J, dim_rep]
            temporal_tokens (torch.Tensor): Temporal tokens of shape [B, T, dim_rep]
        """
        B, T, J, C = x.shape  # [Batch, Time, Joints, Channels]
        if T > self.maxlen:
            raise ValueError(f"Sequence length {T} exceeds maxlen {self.maxlen}")

        x = self.joints_embed(x)  # [B, T, J, dim_feat]
        x = self.pos_drop(x)

        
        x_spatial = x + self.spatial_pos_embed.unsqueeze(1)  # [B, T, J, dim_feat]
        #print(f"After adding spatial_pos_embed Shape: {x_spatial.shape}")  # [64, 50, 20, 256]
        
        # spatial transformer combine batch and time
        x_spatial = x_spatial.view(B * T, J, self.dim_feat)  # [B*T, J, dim_feat]
 
        for blk in self.blocks_spatial:
            x_spatial = blk(x_spatial)      # [B*T, J, dim_feat]
        x_spatial = self.norm(x_spatial)    # [B*T, J, dim_feat]
        x_spatial = x_spatial.view(B, T, J, self.dim_feat)       # back to [B, T, J, dim_feat]
        
        # aggregate over the time dimension to obtain spatial tokens
        spatial_tokens = self.pre_logits(x_spatial.mean(dim=1))  # [B, J, dim_rep]
        #print(f"Spatial Tokens Shape: {spatial_tokens.shape}")  # [64, 20, 256]
        
        x_temporal = x + self.temporal_pos_embed[:, :T, :].unsqueeze(2)         # [B, T, J, dim_feat]
        #print(f"After adding temporal_pos_embed Shape: {x_temporal.shape}")     # [64, 50, 20, 256]
        
        # temporal transformer combine batch and joints
        x_temporal = x_temporal.view(B * J, T, self.dim_feat)  # [B*J, T, dim_feat]

        for blk in self.blocks_temporal:
            x_temporal = blk(x_temporal)        # [B*J, T, dim_feat]
        x_temporal = self.norm(x_temporal)      # [B*J, T, dim_feat]
        x_temporal = x_temporal.view(B, J, T, self.dim_feat)    # back to [B, J, T, dim_feat]
        
        # aggregate over the joints dimension to obtain temporal tokens
        temporal_tokens = self.pre_logits(x_temporal.mean(dim=1))  # [B, T, dim_rep]
        #print(f"Temporal Tokens Shape: {temporal_tokens.shape}")  # [64, 50, 256]

        return spatial_tokens, temporal_tokens  # [B, J=20, dim_rep=256], [B, T=50, dim_rep=256]
