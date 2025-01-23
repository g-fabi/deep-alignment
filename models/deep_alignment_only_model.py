import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from utils.training_utils import init_encoders, init_local_transformer

class DeepAlignmentLoss(nn.Module):
    def __init__(self, modalities, beta=0.5):
        super().__init__()
        self.modalities = modalities
        self.beta = beta

    def forward(self, local_features):
        # Extract spatial and temporal tokens for each modality
        spatial_tokens_mod1 = local_features[self.modalities[0]]['spatial_tokens']  # [bs, n_spatial, dim]
        temporal_tokens_mod1 = local_features[self.modalities[0]]['temporal_tokens']  # [bs, n_temporal, dim]
        spatial_tokens_mod2 = local_features[self.modalities[1]]['spatial_tokens']  # [bs, seq_len, n_joints, dim]
        temporal_tokens_mod2 = local_features[self.modalities[1]]['temporal_tokens']  # [bs, seq_len, n_frames, dim]

        # Collapse seq_len and n_joints/n_frames
        bs, seq_len, n_joints, dim = spatial_tokens_mod2.shape
        spatial_tokens_mod2 = spatial_tokens_mod2.view(bs, seq_len * n_joints, dim)
        n_spatial_mod2 = spatial_tokens_mod2.size(1)

        bs, seq_len, n_frames, dim = temporal_tokens_mod2.shape
        temporal_tokens_mod2 = temporal_tokens_mod2.view(bs, seq_len * n_frames, dim)
        n_temporal_mod2 = temporal_tokens_mod2.size(1)

        # Proceed with cost matrix computation
        C_spatial = self.cost_matrix_batch_torch(spatial_tokens_mod1, spatial_tokens_mod2)
        C_temporal = self.cost_matrix_batch_torch(temporal_tokens_mod1, temporal_tokens_mod2)

        # Compute miu and nu
        bs, n_spatial_mod1, _ = spatial_tokens_mod1.size()
        miu_spatial = torch.ones(bs, n_spatial_mod1).to(C_spatial.device) / n_spatial_mod1
        nu_spatial = torch.ones(bs, n_spatial_mod2).to(C_spatial.device) / n_spatial_mod2

        bs, n_temporal_mod1, _ = temporal_tokens_mod1.size()
        miu_temporal = torch.ones(bs, n_temporal_mod1).to(C_temporal.device) / n_temporal_mod1
        nu_temporal = torch.ones(bs, n_temporal_mod2).to(C_temporal.device) / n_temporal_mod2

        # Compute IPOT distances and combine losses
        loss_spatial = self.IPOT_distance_torch_batch(C_spatial, n_spatial_mod1, n_spatial_mod2, miu_spatial, nu_spatial)
        loss_temporal = self.IPOT_distance_torch_batch(C_temporal, n_temporal_mod1, n_temporal_mod2, miu_temporal, nu_temporal)

        # Combine losses
        total_loss = loss_spatial + loss_temporal
        return total_loss, loss_spatial, loss_temporal

    def cost_matrix_batch_torch(self, x, y):
        """
        Computes the cosine distance between x and y in batches.
        x: [batch_size, n, dim]
        y: [batch_size, m, dim]
        Returns a tensor of shape [batch_size, n, m]
        """
        # Normalize x and y
        x_norm = x / (torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
        y_norm = y / (torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)

        # Compute cosine similarity and convert to distance
        cos_sim = torch.bmm(x_norm, y_norm.transpose(1, 2))  # [batch_size, n, m]
        C = 1 - cos_sim  # Cosine distance
        return C

    def IPOT_distance_torch_batch(self, C, n, m, miu, nu, iteration=20):
        """
        Computes the IPOT distance in batches.
        C: Cost matrix [bs, n, m]
        miu: [bs, n]
        nu: [bs, m]
        Returns a tensor of shape [bs]
        """
        bs = C.size(0)
        T = self.IPOT_torch_batch(C, bs, n, m, miu, nu, iteration)
        cost = torch.einsum('bij,bij->b', C, T)  # Batch-wise trace
        return cost.mean()  # Return positive cost

    def IPOT_torch_batch(self, C, bs, n, m, miu, nu, iteration=20, beta=0.5):
        """
        Computes the IPOT transport plan in batches.
        """
        sigma = torch.ones(bs, m, 1).to(C.device) / m  # [bs, m, 1]
        T = torch.ones(bs, n, m).to(C.device)  # [bs, n, m]
        A = torch.exp(-C / beta)  # [bs, n, m]
        miu = miu.unsqueeze(2)  # [bs, n, 1]
        nu = nu.unsqueeze(2)  # [bs, m, 1]
        for t in range(iteration):
            Q = A * T  # [bs, n, m]
            delta = miu / (torch.bmm(Q, sigma) + 1e-6)
            sigma = nu / (torch.bmm(Q.transpose(1, 2), delta) + 1e-6)
            T = delta * Q * sigma.transpose(1, 2)  # [bs, n, m]
        return T.detach()

class DeepAlignmentModel(LightningModule):
    """
    A model for deep alignment training using only the Deep Alignment Loss.
    """
    def __init__(
        self,
        modalities,
        encoders,
        local_transformers,
        optimizer_name='adam',
        lr=0.001,
        **kwargs
    ):
        super().__init__()
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)
        self.local_transformers = nn.ModuleDict(local_transformers)
        self.deep_alignment_loss_fn = DeepAlignmentLoss(modalities)
        self.optimizer_name = optimizer_name
        self.lr = lr

    def forward(self, batch):
        for m in self.modalities:
            batch[m] = batch[m].to(self.device)

        global_features = {}
        local_features = {}

        for m in self.modalities:
            # Use the local transformer only
            _, local_feat = self.local_transformers[m](batch[m])
            local_features[m] = local_feat

        return global_features, local_features

    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()

        global_features, local_features = self(batch)

        loss_deep_alignment, loss_spatial, loss_temporal = self.deep_alignment_loss_fn(local_features)
        self.log("train_loss", loss_deep_alignment, on_step=True, on_epoch=True)
        self.log("train_loss_spatial", loss_spatial, on_step=True, on_epoch=True)
        self.log("train_loss_temporal", loss_temporal, on_step=True, on_epoch=True)
        return loss_deep_alignment

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()

        global_features, local_features = self(batch)
        loss_deep_alignment, loss_spatial, loss_temporal = self.deep_alignment_loss_fn(local_features)
        self.log("val_loss", loss_deep_alignment, on_step=True, on_epoch=True)
        self.log("val_loss_spatial", loss_spatial, on_step=True, on_epoch=True)
        self.log("val_loss_temporal", loss_temporal, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer 