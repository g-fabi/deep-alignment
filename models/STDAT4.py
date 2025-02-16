import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.checkpoint

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std)

class Block(nn.Module):
    def __init__(self, d_model, num_heads, dropout, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )

    
    def forward(self, x):
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x2 = self.mlp(self.norm2(x))
        return x + x2

class STDAT(nn.Module):
    def __init__(self, in_channels=6, sample_length=50, d_model=256,
                 dim_rep=256, depth=5, num_heads=8, dropout=0.1, mlp_ratio=4.0, out_size=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.dim_rep = dim_rep
        self.sample_length = sample_length

        # Projection maps IMU features to d_model (temporal branch)
        self.input_projection = nn.Linear(in_channels, d_model)
        self.pos_drop = nn.Dropout(p=dropout)

        # Separate CLS tokens for each branch
        self.cls_token_temporal = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token_spatial = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_token_temporal, std=.02)
        trunc_normal_(self.cls_token_spatial, std=.02)

        # Positional Embeddings for each branch
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.sample_length + 1, d_model))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, in_channels + 1, d_model))
        trunc_normal_(self.temporal_pos_embed, std=.02)
        trunc_normal_(self.spatial_pos_embed, std=.02)
        
        # NEW: Spatial branch input projection
        self.spatial_input_projection = nn.Linear(self.sample_length, d_model)  # projects each sensor's T values
        
        self.out_size = out_size if out_size is not None else dim_rep

        self.blocks_temporal = nn.ModuleList([
            Block(d_model, num_heads, dropout, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.blocks_spatial = nn.ModuleList([
            Block(d_model, num_heads, dropout, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.norm_temporal = nn.LayerNorm(d_model)
        self.norm_spatial = nn.LayerNorm(d_model)
        
        # Global fusion layer to combine branch outputs
        self.global_fuse = nn.Linear(2 * d_model, d_model)

        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(d_model, dim_rep)),
            ('act', nn.ReLU())
        ]))

        # Add final projection layer to match expected output size
        self.final_projection = nn.Sequential(
            nn.LayerNorm(dim_rep),
            nn.Linear(dim_rep, self.out_size)
        )

        self.apply(self._init_weights)
        self.__dict__['stdat'] = self
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        #print("STDAT forward: initial x shape:", x.shape)
        x = x.transpose(1, 2)
        #print("After transpose, x shape:", x.shape)
        B, T, C = x.shape
        x_proj = self.input_projection(x)  # [B, T, d_model]
        #print("After input_projection, x_proj shape:", x_proj.shape)
        x_proj = self.pos_drop(x_proj)
        
        cls_temp = self.cls_token_temporal.expand(B, 1, -1)
        #print("CLS token temporal shape:", cls_temp.shape)
        x_temporal = torch.cat((cls_temp, x_proj), dim=1)  # [B, T+1, d_model]
        #print("After concatenation, x_temporal shape:", x_temporal.shape)
        #print("temporal_pos_embed shape:", self.temporal_pos_embed.shape)
        x_temporal = x_temporal + self.temporal_pos_embed[:, :T+1, :]
        #print("After adding temporal_pos_embed, x_temporal shape:", x_temporal.shape)
        x_temporal = self.pos_drop(x_temporal)
        for blk in self.blocks_temporal:
            x_temporal = torch.utils.checkpoint.checkpoint(blk, x_temporal)
        x_temporal = self.norm_temporal(x_temporal)
        temporal_global = x_temporal[:, 0, :]  # dedicated CLS token output


        # Spatial Branch: (treat sensor channels as tokens)
        # x transpose: now [B, C, T]
        x_spatial = x.transpose(1, 2)
        x_spatial = self.spatial_input_projection(x_spatial)  # [B, C, d_model]
        cls_spat = self.cls_token_spatial.expand(B, 1, -1)
        x_spatial = torch.cat((cls_spat, x_spatial), dim=1)  # [B, C+1, d_model]
        x_spatial = x_spatial + self.spatial_pos_embed[:, : (x_spatial.shape[1]), :]
        x_spatial = self.pos_drop(x_spatial)
        for blk in self.blocks_spatial:
            x_spatial = torch.utils.checkpoint.checkpoint(blk, x_spatial)
        x_spatial = self.norm_spatial(x_spatial)
        spatial_global = x_spatial[:, 0, :]  # dedicated CLS token

        # Global fusion: fuse temporal and spatial global tokens
        fused = torch.cat([temporal_global, spatial_global], dim=-1)  # [B, 2*d_model]
        fused_global = self.global_fuse(fused)
        global_features = self.pre_logits(fused_global)

        # Add final projection to match expected output size
        global_features = self.final_projection(global_features)
            
        local_features = {
            'temporal_tokens': x_temporal[:, 1:, :],  # [B, T, d_model]
            'spatial_tokens': x_spatial[:, 1:, :]       # [B, C, d_model]
        }
        return global_features, local_features