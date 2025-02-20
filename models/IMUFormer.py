import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, k, d_model, seq_len):
        super().__init__()
        
        self.embedding = nn.Parameter(torch.zeros([k, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(seq_len)], dtype=torch.float, requires_grad=False) \
                             .unsqueeze(1).repeat(1, k)
        s = 0.0
        interval = seq_len / k
        mu_list = []
        for _ in range(k):
            mu_list.append(torch.tensor(s, dtype=torch.float))
            s = s + interval
        self.mu = nn.Parameter(torch.stack(mu_list).unsqueeze(0))
        self.sigma = nn.Parameter(torch.stack([
            torch.tensor(50.0, dtype=torch.float, requires_grad=True)
            for _ in range(k)
        ]).unsqueeze(0))
        
    def normal_pdf(self, pos, mu, sigma):
        a = pos - mu
        log_p = -1*torch.mul(a, a)/(2*(sigma**2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        device = inputs.device
        pdfs = self.normal_pdf(self.positions.to(device), self.mu.to(device), self.sigma.to(device))
        pos_enc = torch.matmul(pdfs, self.embedding)
        
        return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, _heads, dropout, seq_len):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(d_model, heads)
        self._attention = nn.MultiheadAttention(seq_len, _heads)
        
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.cnn_units = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_units, (1, 1)),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, self.cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src is assumed to be (B, T, d_model)

        # First branch: standard attention
        # Transpose to (T, B, d_model)
        src_t = src.transpose(0, 1)
        attn_out1 = self.attention(src_t, src_t, src_t)[0]  # (T, B, d_model)
        attn_out1 = attn_out1.transpose(0, 1)               # (B, T, d_model)

        # Second branch: attention on transposed tokens
        # Compute src2 = src.transpose(-1, -2) -> (B, d_model, T)
        # Then bring batch dimension to second position: (d_model, B, T)
        src2 = src.transpose(-1, -2).transpose(0, 1)
        attn_out2 = self._attention(src2, src2, src2)[0]    # (d_model, B, T)
        # Reverse the transpositions: first transpose back -> (B, d_model, T), then swap last two dims -> (B, T, d_model)
        attn_out2 = attn_out2.transpose(0, 1).transpose(-1, -2)

        # Sum the outputs and apply normalization
        src = self.attn_norm(src + attn_out1 + attn_out2)
        src = self.final_norm(src + self.cnn(src.unsqueeze(dim=1)).squeeze(dim=1))
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, _heads, seq_len, num_layer=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(TransformerEncoderLayer(d_model, heads, _heads, dropout, seq_len))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src
    
class IMUFormer(nn.Module):
    def __init__(self, d_model, num_layers, heads, _heads, sample_length, in_channels, dropout=0.1, **kwargs):
        """
        Args:
            d_model: Embedding dimension.
            num_layers: Number of transformer encoder layers for each stream.
            heads, _heads: Attention heads configuration.
            sample_length: Number of time steps.
            in_channels: Number of inertial channels.
            dropout: Dropout probability.
            **kwargs: Extra keyword arguments (e.g. out_size) from dataset config merger.
        """
        super().__init__()
        self.in_channels = in_channels  # store the expected number of channels

        # Projection for temporal branch: project raw inertial input (in_channels)
        # to the model dimension (d_model) prior to adding positional encoding.
        self.temporal_proj = nn.Linear(in_channels, d_model)
        
        #  Temporal Stream 
        self.temporal_pos_encoding = PositionalEncoding(k=20, d_model=d_model, seq_len=sample_length)
        self.temporal_transformer = TransformerEncoder(d_model, heads, _heads, sample_length, num_layer=num_layers, dropout=dropout)
        
        # Spatial Stream 
        self.channel_embedding = nn.Linear(1, d_model)
        self.spatial_pos_encoding = PositionalEncoding(k=20, d_model=d_model, seq_len=in_channels)
        self.spatial_transformer = TransformerEncoder(d_model, heads, _heads, in_channels, num_layer=num_layers, dropout=dropout)
        
        # Global Feature 
        self.global_fc = nn.Linear(2 * d_model, d_model)
        out_size = kwargs.get("out_size", d_model)
        self.out_fc = nn.Linear(d_model, out_size)
        self.out_size = out_size

        # Force all parameters of this encoder to be float32 
        self.apply(lambda m: m.float())
        self.__dict__['imuformer'] = self
        
    def forward(self, x):
        """
        Args:
            x: Inertial input tensor of shape [B, sample_length, num_channels]
        Returns:
            global_feature: Aggregated global feature.
            spatial_tokens: Local spatial tokens (one per channel).
            temporal_tokens: Local temporal tokens (one per time step).
        """
        # Ensure x is in shape [B, sample_length, in_channels]
        if x.shape[-1] != self.in_channels:
            x = x.transpose(1, 2)
        # Ensure input tensor dtype matches the layer weights (float32)
        x = x.to(self.temporal_proj.weight.dtype)
        B, T, C = x.shape
        
        # Temporal Stream 
        # Project x from in_channels to d_model before adding positional encoding.
        x_temp = self.temporal_proj(x)                           # [B, T, d_model]
        temporal_input = self.temporal_pos_encoding(x_temp)      # [B, T, d_model]
        temporal_tokens = self.temporal_transformer(temporal_input)  # [B, T, d_model]
        temporal_global = temporal_tokens.mean(dim=1)              # [B, d_model]
        
        # Spatial Stream 
        x_spatial = x.transpose(1, 2)  # [B, C, T]
        x_spatial = x_spatial.mean(dim=2, keepdim=True)  # [B, C, 1]
        x_spatial = x_spatial.mean(dim=2, keepdim=True)  # [B, C, 1]
        x_spatial = self.channel_embedding(x_spatial)    # [B, C, d_model]
        spatial_input = self.spatial_pos_encoding(x_spatial)   # [B, C, d_model]
        spatial_input = self.spatial_pos_encoding(x_spatial)   # [B, C, d_model]
        spatial_tokens = self.spatial_transformer(spatial_input) # [B, C, d_model]
        spatial_global = spatial_tokens.mean(dim=1)              # [B, d_model]
        
        # Global Feature 
        global_cat = torch.cat([temporal_global, spatial_global], dim=-1)  # [B, 2*d_model]
        global_feature = self.global_fc(global_cat)  # [B, d_model]
        global_feature = self.out_fc(global_feature) # Final global feature
        
        return global_feature, spatial_tokens, temporal_tokens

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False