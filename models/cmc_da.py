import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from models.mlp import ProjectionMLP
from models.deep_alignment import DeepAlignmentLoss

class MM_NTXent(LightningModule):
    """
    Multimodal adaptation of NTXent loss as a LightningModule.
    Computes the cosine similarity matrix between the two modalities' 
    global features and builds the contrastive loss.
    """
    def __init__(self, batch_size, modalities, temperature=0.1):
        super().__init__()
        self.save_hyperparameters()  # saves batch_size, modalities, temperature
        self.batch_size = batch_size
        self.modalities = modalities
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def get_cosine_sim_matrix(self, features_1, features_2):
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        return torch.matmul(features_1, features_2.T)

    def forward(self, features):
        m1, m2 = self.modalities[0], self.modalities[1]
        f1 = features[m1]
        f2 = features[m2]
        sim_matrix = self.get_cosine_sim_matrix(f1, f2)
        
        # Positive pairs: diagonal elements.
        pos = sim_matrix.diag()  # shape: (batch_size,)
        pos = torch.cat([pos, pos], dim=0)
        
        # Negative pairs: all off-diagonals.
        mask = torch.eye(self.batch_size, dtype=torch.bool, device=sim_matrix.device)
        neg1 = sim_matrix[~mask].view(self.batch_size, -1)
        neg2 = sim_matrix.T[~mask].view(self.batch_size, -1)
        negatives = torch.cat([neg1, neg2], dim=0)
        
        # Construct logits, placing positive similarity as the first column.
        logits = torch.cat([pos.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long, device=logits.device)
        logits = logits / self.temperature
        
        loss = self.criterion(logits, labels)
        pos_mean = pos.mean()
        neg_mean = negatives.mean()
        return loss, pos_mean, neg_mean

class ContrastiveMultiviewCodingDA(LightningModule):
    """
    Combined framework extending CMC with Deep Alignment.
    For each modality, it uses:
      • A projection head (ProjectionMLP) on the flattened (global) encoder output
        for the NTXent contrastive loss.
      • A deep alignment branch that extracts spatial and temporal tokens from the encoder output.
    
    The final training loss is:
         total_loss = ntxent_loss + lambda_da * deep_alignment_loss
    """
    def __init__(self, modalities, encoders, hidden=[256, 128], batch_size=64, temperature=0.1, 
                 optimizer_name='adam', lr=0.001, lambda_da=1.0, init_log_sigma_ntxent=0.0, init_log_sigma_da=0.0, da_kwargs=None):
        """
        Args:
          modalities: List of modality names
          encoders: Dictionary of pretrained encoders.
          hidden: Hidden layer configuration for ProjectionMLP.
          batch_size: Batch size used for contrastive loss.
          temperature: Temperature for NTXent loss.
          optimizer_name: Optimizer name.
          lr: Learning rate.
          lambda_da: Weight to balance the deep alignment loss.
          da_kwargs: Optional kwargs for DeepAlignmentLoss (e.g., weight_spatial, beta, iteration)
        """
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'temperature', 'optimizer_name', 'lr', 'lambda_da', 'init_log_sigma_ntxent', 'init_log_sigma_da')
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)
        
        # Ensure that each encoder is in float32
        for m in self.modalities:
            self.encoders[m].float()
        
        # Projection heads for global features to be used in NTXent loss.
        projections = {}
        for m in modalities:
            projections[m] = ProjectionMLP(in_size=self.encoders[m].out_size, hidden=hidden)
        self.projections = nn.ModuleDict(projections)
        
        self.ntxent_loss = MM_NTXent(batch_size=batch_size, modalities=modalities, temperature=temperature)
        
        # Deep Alignment loss for token alignment
        if da_kwargs is None:
            da_kwargs = {}
        self.da_loss = DeepAlignmentLoss(**da_kwargs)
        
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.lambda_da = lambda_da
        
        # Learnable loss weighting parameters
        # Initialize log_sigma_ntxent and log_sigma_da to 0 so that exp(0)=1 initially.
        self.log_sigma_ntxent = nn.Parameter(torch.tensor(init_log_sigma_ntxent))
        self.log_sigma_da = nn.Parameter(torch.tensor(init_log_sigma_da))

    def _forward_modality(self, modality, x):
        out = self.encoders[modality](x)
        
        # Global feature for NTXent
        if isinstance(out, (tuple, list)):
            global_feat = out[0]
        else:
            global_feat = out
        global_feat = torch.flatten(global_feat, start_dim=1)
        proj_global = self.projections[modality](global_feat)
        
        # Extract spatial & temporal tokens for deep alignment.
        if isinstance(out, (tuple, list)):
            if len(out) >= 3:
                spatial = out[1]
                temporal = out[2]
            elif len(out) == 2 and isinstance(out[1], dict):
                spatial = out[1].get("spatial") or out[1].get("spatial_skeleton")
                temporal = out[1].get("temporal") or out[1].get("temporal_skeleton")
            else:
                raise ValueError("Encoder output format not supported for Deep Alignment")
        elif isinstance(out, dict):
            spatial = out.get("spatial")
            temporal = out.get("temporal")
        else:
            raise ValueError("Encoder output format not supported for Deep Alignment")
        return proj_global, spatial, temporal

    def forward(self, batch):
        """
        Returns dictionaries for:
          • Global features (for NTXent)
          • Spatial tokens
          • Temporal tokens
        """
        global_feats = {}
        spatial_tokens = {}
        temporal_tokens = {}
        for m in self.modalities:
            proj_global, spatial, temporal = self._forward_modality(m, batch[m])
            global_feats[m] = proj_global
            spatial_tokens[m] = spatial
            temporal_tokens[m] = temporal
        return global_feats, spatial_tokens, temporal_tokens

    def training_step(self, batch, batch_idx):
        global_feats = {}
        spatial_tokens = {}
        temporal_tokens = {}
        for m in self.modalities:
            proj_global, spatial, temporal = self._forward_modality(m, batch[m])
            global_feats[m] = proj_global
            spatial_tokens[m] = spatial
            temporal_tokens[m] = temporal

        loss_ntxent, pos_mean, neg_mean = self.ntxent_loss(global_feats)
        
        m1, m2 = self.modalities[0], self.modalities[1]
        loss_da, loss_spatial, loss_temporal = self.da_loss(
            spatial_tokens[m1], spatial_tokens[m2], temporal_tokens[m1], temporal_tokens[m2]
        )
        
        # Compute weighted losses using uncertainty weighting
        weighted_ntxent = torch.exp(-self.log_sigma_ntxent) * loss_ntxent + self.log_sigma_ntxent
        weighted_da = torch.exp(-self.log_sigma_da) * loss_da + self.log_sigma_da
        total_loss = weighted_ntxent + weighted_da

        self.log("log_sigma_ntxent", self.log_sigma_ntxent)
        self.log("log_sigma_da", self.log_sigma_da)
        self.log("ssl_train_loss", total_loss)
        self.log("cmc_loss", loss_ntxent)
        self.log("da_loss", loss_da)
        self.log("da_loss_spatial", loss_spatial)
        self.log("da_loss_temporal", loss_temporal)
        self.log("avg_positive_sim", pos_mean)
        self.log("avg_neg_sim", neg_mean)
        return total_loss

    def validation_step(self, batch, batch_idx):
        global_feats = {}
        spatial_tokens = {}
        temporal_tokens = {}
        for m in self.modalities:
            proj_global, spatial, temporal = self._forward_modality(m, batch[m])
            global_feats[m] = proj_global
            spatial_tokens[m] = spatial
            temporal_tokens[m] = temporal

        loss_ntxent, _, _ = self.ntxent_loss(global_feats)
        m1, m2 = self.modalities[0], self.modalities[1]
        loss_da, loss_spatial, loss_temporal = self.da_loss(
            spatial_tokens[m1], spatial_tokens[m2], temporal_tokens[m1], temporal_tokens[m2]
        )
        
        total_loss = loss_ntxent + self.lambda_da * loss_da
        self.log("ssl_val_loss", total_loss)
        self.log("cmc_val_loss", loss_ntxent)
        self.log("da_val_loss", loss_da)
        self.log("da_val_loss_spatial", loss_spatial)
        self.log("da_val_loss_temporal", loss_temporal)
        return total_loss

    def configure_optimizers(self):
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Only Adam is supported at the moment")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "ssl_train_loss"}
        } 