## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from einops import rearrange, repeat

import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.drop import DropPath

def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Output dimension is dim*2 because we split it into keys and values
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x: [B, N_q, C] -> queries
        # y: [B, N_k, C] -> keys and values
        B, N_q, C = x.shape
        B, N_k, _ = y.shape  # Use y's sequence length here
        # Compute queries and reshape:
        q = self.q(x).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # Compute keys and values. Note: output will have shape [B, N_k, 2 * C]
        kv = self.kv(y).reshape(B, N_k, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # k,v: [B, num_heads, N_k, head_dim]

        # Compute attention scores using q and k
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_q, N_k]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Weighted sum over v's: result shape [B, num_heads, N_q, head_dim]
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.attn2 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        y_ = self.attn1(self.norm1(y))
        y = y + self.drop_path(y_)
        y = y + self.drop_path(self.mlp1(self.norm2(y)))

        x = x + self.drop_path(self.attn2(self.norm3(x)))
        x = x + self.cross_attn(x, y_)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        return x, y


class MixedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)

class PoseFormer(nn.Module):
    def __init__(self, sample_length, n_joints, input_channels, opt, output_size=None,
                 embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=None, **kwargs):
        """
        Args:
            sample_length (int, tuple): input frame number (sample length)
            n_joints (int, tuple): number of joints
            input_channels (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            output_size (int, tuple): final output dimension (default: n_joints * 3)
        """
        super().__init__()
        from argparse import Namespace
        if isinstance(opt, dict):
            opt = Namespace(**opt)
        
        # print(f"\nPoseFormerV3 Init:")
        # print(f"input_channels: {input_channels}")
        # print(f"embed_dim_ratio: {embed_dim_ratio}")
        # print(f"num_joints: {n_joints}")

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        depth = opt.layers
        embed_dim = embed_dim_ratio * n_joints   
        self.embed_dim = embed_dim  # Store for classifier
        self.dim_rep = embed_dim * 2  # Global feature dimension
        if output_size is None:
            out_dim = n_joints * 3     
        else:
            out_dim = output_size
        self.num_frame_kept = opt.number_of_kept_frames
        self.num_coeff_kept = opt.number_of_kept_coeffs if opt.number_of_kept_coeffs is not None else self.num_frame_kept

        #print(f"Joint_embedding input dim: {input_channels}, output dim: {embed_dim_ratio}")
        self.Joint_embedding = nn.Linear(input_channels, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(input_channels*n_joints, embed_dim)
        
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, n_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        
        # Add CLS tokens (optional but helps with alignment)
        self.cls_token_spatial = nn.Parameter(torch.zeros(1, 1, embed_dim_ratio))
        self.cls_token_temporal = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token_spatial, std=.02)
        trunc_normal_(self.cls_token_temporal, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Keep original transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.Spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        if opt.naive:
            self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                MixedBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.weighted_mean = torch.nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
        self.weighted_mean_ = torch.nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, out_dim)
        )

        # Add cross-attention block for global feature fusion
        self.cross_block = CrossBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate
        )

        self._out_size = out_dim
        self.__dict__['poseformer'] = self
        
        if sample_length is not None:
            self.num_frame_kept = sample_length

    def Spatial_forward_features(self, x):
        """Process spatial features through transformer blocks.
        
        Args:
            x: Input tensor of shape [B, T, J, C]
        Returns:
            tuple: (spatial_tokens, cls_spatial)
                - spatial_tokens: [B, T, J, embed_dim_ratio]
                - cls_spatial: [B, T, 1, embed_dim_ratio]
        """
        # print("\nSpatial_forward_features Debug:")
        # print(f"Input x shape: {x.shape}")
        # print(f"Expected dims: [batch_size, num_frames, num_joints, in_chans]")

        b, f, p, _ = x.shape  
        num_frame_kept = self.num_frame_kept

        start = (f - num_frame_kept) // 2
        index = torch.arange(start, start + num_frame_kept, device=x.device)
        # print(f"Selecting frames {start} to {start + num_frame_kept} from {f} frames")

        # x[:, index] has shape [B, num_frame_kept, p, C]
        y = x[:, index]
        # print(f"After frame selection, y shape: {y.shape}")

        x_flat = y.reshape(b * num_frame_kept * p, y.shape[-1])
        # print(f"After flattening, x_flat shape: {x_flat.shape}")
        # print(f"Joint_embedding weight shape: {self.Joint_embedding.weight.shape}")
        # print(f"Joint_embedding expects input dim: {self.Joint_embedding.in_features}")

        # Apply Joint_embedding to each joint vector
        x_emb = self.Joint_embedding(x_flat)
        # print(f"After Joint_embedding, x_emb shape: {x_emb.shape}")

        # Reshape back to [B * num_frame_kept, p, embed_dim_ratio]
        x = x_emb.reshape(b * num_frame_kept, p, -1)
        # print(f"Final reshape, x shape: {x.shape}\n")

        # Add CLS token to spatial stream
        cls_tokens = self.cls_token_spatial.expand(b*num_frame_kept, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings only to joint tokens (skip the CLS token)
        x[:, 1:, :] += self.Spatial_pos_embed
        x = self.pos_drop(x)

        # Process through spatial blocks
        for blk in self.Spatial_blocks:
            x = checkpoint(blk, x)

        x = self.Spatial_norm(x)
        
        # Split CLS token and spatial tokens
        cls_spatial = x[:, 0:1, :]  # Keep CLS token
        spatial_tokens = x[:, 1:, :]  # Keep spatial tokens
        
        # Reshape back to include frame dimension
        spatial_tokens = rearrange(spatial_tokens, '(b f) p c -> b f p c', f=num_frame_kept)
        cls_spatial = rearrange(cls_spatial, '(b f) n c -> b f n c', f=num_frame_kept)
        
        return spatial_tokens, cls_spatial

    def forward_features(self, x, spatial_tokens, cls_spatial):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept

        # DCT transform and frequency embedding
        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, num_coeff_kept, -1)
        x = self.Freq_embedding(x)

        # Add CLS token to temporal stream
        cls_tokens = self.cls_token_temporal.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings only to frequency tokens (skip the CLS token)
        x[:, 1:, :] += self.Temporal_pos_embed_
        # Flatten spatial tokens from [B, f, J, embed_dim_ratio] to [B, f, J*embed_dim_ratio]
        spatial_tokens = spatial_tokens.flatten(2)
        spatial_tokens += self.Temporal_pos_embed
        
        # Concatenate streams
        x = torch.cat((x, spatial_tokens), dim=1)

        # Process through temporal blocks
        for blk in self.blocks:
            x = checkpoint(blk, x)

        x = self.Temporal_norm(x)
        
        # Split temporal features
        cls_temporal = x[:, 0:1, :]  # CLS token
        temporal_tokens = x[:, 1:, :]  # Rest of tokens
        
        # Store pure tokens before mixing
        pure_temporal_tokens = temporal_tokens.clone()
        pure_spatial_tokens = spatial_tokens.clone()
        
        # Use CrossBlock for enhanced interaction
        temporal_tokens, spatial_tokens = self.cross_block(
            temporal_tokens,  # temporal stream
            spatial_tokens    # spatial stream
        )
        
        # Global feature generation with enhanced tokens
        global_feat = torch.cat((
            self.weighted_mean(temporal_tokens[:, :self.num_coeff_kept]),
            self.weighted_mean_(spatial_tokens)
        ), dim=-1)
        
        return global_feat, {
            'spatial_tokens': pure_spatial_tokens,
            'temporal_tokens': pure_temporal_tokens[:, 1:]
        }

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, T, J, 1]
        Returns:
            tuple: (global_features, local_features)
                - global_features: Tensor of shape [B, embed_dim*2]
                - local_features: Dict containing:
                    - 'spatial_tokens': [B, T, J, embed_dim_ratio]
                    - 'temporal_tokens': [B, T+F, embed_dim]
        """
        x = x.squeeze(-1)
        # If input is [B, C, T, J] (C==3), permute to [B, T, J, C]
        if x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        x_ = x.clone()  # Use the correctly permuted tensor for frequency branch

        spatial_tokens, cls_spatial = self.Spatial_forward_features(x)
        global_feat, local_features = self.forward_features(x_, spatial_tokens, cls_spatial)
        global_feat = global_feat.view(x.shape[0], -1)
        
        # Add final projection through head
        global_feat = self.head(global_feat)
            
        return global_feat, local_features

    @property
    def out_size(self):
        return self._out_size

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False 