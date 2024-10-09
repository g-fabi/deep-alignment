# models/STDAT3.py

import torch
import torch.nn as nn
from collections import OrderedDict

def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std)

class Block(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [Seq Length, Batch, d_model]
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x2 = self.mlp(self.norm2(x))
        x = x + x2
        return x

class STDAT(nn.Module):
    def __init__(self, imu_feature_count=6, max_seq_len=50, d_model=256,
                 dim_rep=256, depth=5, num_heads=8, dropout=0.1):
        super(STDAT, self).__init__()
        self.imu_feature_count = imu_feature_count
        self.d_model = d_model
        self.dim_rep = dim_rep
        self.max_seq_len = max_seq_len

        # projection maps imu features to d_model
        self.input_projection = nn.Linear(imu_feature_count, d_model)
        self.pos_drop = nn.Dropout(p=dropout)

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, imu_feature_count, d_model))
        trunc_normal_(self.temporal_pos_embed, std=.02)
        trunc_normal_(self.spatial_pos_embed, std=.02)

        self.blocks_temporal = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.blocks_spatial = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm_temporal = nn.LayerNorm(d_model)
        self.norm_spatial = nn.LayerNorm(d_model)
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(d_model, dim_rep)),
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
        Forward pass for STDAT.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C=6]

        Returns:
            spatial_tokens (torch.Tensor): Spatial tokens of shape [B, C=6, dim_rep]
            temporal_tokens (torch.Tensor): Temporal tokens of shape [B, T, dim_rep]
        """
        # x: [B, T, C=6]
        B, T, C = x.shape
        #print(f"STDAT Forward Input Shape: {x.shape}")  # [64,50,6]
        
        x_proj = self.input_projection(x)  # [B, T, dim_rep]
        #print(f"After input_projection x_proj Shape: {x_proj.shape}")  # [64,50,256]
        x_proj = self.pos_drop(x_proj)

        x_temporal = x_proj + self.temporal_pos_embed[:, :T, :]  # [B, T, 256]
        #print(f"After adding temporal_pos_embed Shape: {x_temporal.shape}")  # [64,50,256]
        x_temporal = x_temporal.permute(1, 0, 2)  # [T, B, 256]
        for blk in self.blocks_temporal:
            x_temporal = blk(x_temporal)  # [T, B, 256]
        x_temporal = x_temporal.permute(1, 0, 2)  # [B, T, 256]
        x_temporal = self.norm_temporal(x_temporal)
        temporal_tokens = self.pre_logits(x_temporal)  # [B, T, 256]
        #print(f"Temporal Tokens Shape: {temporal_tokens.shape}")  # [64,50,256]

        # Aggregate over the temporal dimension
        x_spatial = x_proj.mean(dim=1)  # [B, 256]
        #print(f"After averaging over T, x_spatial Shape: {x_spatial.shape}")  # [64,256]
        
        # Expand to [B, C=6, dim_rep] by repeating
        x_spatial = x_spatial.unsqueeze(1).repeat(1, C, 1)  # [B,6,256]

        x_spatial = x_spatial + self.spatial_pos_embed  # [B,6,256]
        #print(f"After adding spatial_pos_embed Shape: {x_spatial.shape}")  # [64,6,256]
        
        x_spatial = x_spatial.permute(1, 0, 2).contiguous()  # [6,64,256]
        for blk in self.blocks_spatial:
            x_spatial = blk(x_spatial)  # [6,64,256]
        
        x_spatial = x_spatial.permute(1, 0, 2).contiguous()  # [64,6,256]
        x_spatial = self.norm_spatial(x_spatial)  # [64,6,256]
        spatial_tokens = self.pre_logits(x_spatial)  # [64,6,256]
        #print(f"Spatial Tokens Shape: {spatial_tokens.shape}")  # [64,6,256]

        return spatial_tokens, temporal_tokens  # [64,6,256], [64,50,256]
