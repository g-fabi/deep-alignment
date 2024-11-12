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

        # Projection maps IMU features to d_model
        self.input_projection = nn.Linear(imu_feature_count, d_model)
        self.pos_drop = nn.Dropout(p=dropout)

        # Learnable [CLS] tokens for temporal and spatial transformers
        self.cls_token_temporal = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token_spatial = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_token_temporal, std=.02)
        trunc_normal_(self.cls_token_spatial, std=.02)

        # Positional Embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, imu_feature_count + 1, d_model))
        trunc_normal_(self.temporal_pos_embed, std=.02)
        trunc_normal_(self.spatial_pos_embed, std=.02)
        
        self.out_size = dim_rep

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
        print(f"x device: {x.device}")
        x = x.to(self.cls_token_temporal.device)
        """
        Forward pass for STDAT.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C=6]

        Returns:
            global_features (torch.Tensor): Global features of shape [B, dim_rep]
            local_features (dict): Local features including spatial and temporal tokens
        """
        # x: [B, T, C=6]
        print(f"x shape: {x.shape}")
        B, T, C = x.shape
        x_proj = self.input_projection(x)  # [B, T, d_model]
        x_proj = self.pos_drop(x_proj)

        # Temporal Processing
        cls_tokens_temporal = self.cls_token_temporal.expand(B, -1, -1)  # [B, 1, d_model]
        x_temporal = torch.cat((cls_tokens_temporal, x_proj), dim=1)  # [B, T+1, d_model]

        # Add positional embeddings
        x_temporal = x_temporal + self.temporal_pos_embed[:, :T + 1, :]  # [B, T+1, d_model]
        x_temporal = self.pos_drop(x_temporal)

        # Prepare for transformer blocks: [Seq Length, Batch, d_model]
        x_temporal = x_temporal.permute(1, 0, 2)  # [T+1, B, d_model]
        for blk in self.blocks_temporal:
            x_temporal = blk(x_temporal)  # [T+1, B, d_model]
        x_temporal = x_temporal.permute(1, 0, 2).contiguous()  # [B, T+1, d_model]
        x_temporal = self.norm_temporal(x_temporal)

        # Extract [CLS] token output for temporal global features
        cls_output_temporal = x_temporal[:, 0, :]  # [B, d_model]
        temporal_global = self.pre_logits(cls_output_temporal)  # [B, dim_rep]

        # Extract temporal local features (excluding [CLS] token)
        temporal_tokens = x_temporal[:, 1:, :]  # [B, T, d_model]

        # Spatial Processing
        # Prepare spatial input
        x_spatial = self.input_projection(x.mean(dim=1))  # [B, d_model]
        x_spatial = x_spatial.unsqueeze(1)  # [B, 1, d_model]
        cls_tokens_spatial = self.cls_token_spatial.expand(B, -1, -1)  # [B, 1, d_model]
        print(f"cls_tokens_spatial shape: {cls_tokens_spatial.shape}")  # [B, 1, d_model]
        print(f"x_spatial shape: {x_spatial.shape}")  # [B, 1, d_model]
        x_spatial = torch.cat((cls_tokens_spatial, x_spatial), dim=1)  # [B, 2, d_model]
        x_spatial = x_spatial + self.spatial_pos_embed[:, :x_spatial.shape[1], :]  # [B, 2, d_model]
        x_spatial = self.pos_drop(x_spatial)

        # Spatial transformer blocks
        x_spatial = x_spatial.permute(1, 0, 2).contiguous()  # [2, B, d_model]
        for blk in self.blocks_spatial:
            x_spatial = blk(x_spatial)  # [2, B, d_model]
        x_spatial = x_spatial.permute(1, 0, 2).contiguous()  # [B, 2, d_model]
        x_spatial = self.norm_spatial(x_spatial)

        # Extract [CLS] token output for spatial global features
        cls_output_spatial = x_spatial[:, 0, :]  # [B, d_model]
        spatial_global = self.pre_logits(cls_output_spatial)  # [B, dim_rep]

        # Extract spatial local features (excluding [CLS] token)
        spatial_tokens = x_spatial[:, 1:, :]  # [B, 1, d_model]

        global_features = temporal_global + spatial_global  # [B, dim_rep]

        local_features = {
            'temporal_tokens': temporal_tokens,  # [B, T, d_model]
            'spatial_tokens': spatial_tokens     # [B, 1, d_model]
        }

        return global_features, local_features
