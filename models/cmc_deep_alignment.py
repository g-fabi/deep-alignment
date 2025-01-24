import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from models.mlp import ProjectionMLP
from models.deep_alignment_only_model import DeepAlignmentLoss
from models.cmc import MM_NTXent

class ContrastiveMultiviewCodingWithDeepAlignment(LightningModule):
    """
    Combined model for CMC and Deep Alignment losses.
    """
    def __init__(
        self, 
        modalities, 
        global_encoders, 
        local_encoders, 
        hidden_cmc, 
        hidden_da, 
        lr_cmc, 
        lr_da, 
        batch_size, 
        temperature, 
        optimizer_name_ssl, 
        beta=0.5
    ):
        super().__init__()

        self.modalities = modalities
        self.global_encoders = nn.ModuleDict(global_encoders)
        self.local_encoders = nn.ModuleDict(local_encoders)

        self.hidden_cmc = hidden_cmc
        self.hidden_da = hidden_da
        self.lr_cmc = lr_cmc
        self.lr_da = lr_da
        self.batch_size = batch_size
        self.temperature = temperature
        self.optimizer_name_ssl = optimizer_name_ssl or 'adam'
        self.beta = beta

        self.save_hyperparameters(ignore=['global_encoders', 'local_encoders'])

        # Projections for CMC
        self.global_projections = nn.ModuleDict({
            m: ProjectionMLP(in_size=self.global_encoders[m].out_size, hidden=self.hidden_cmc)
            for m in self.modalities
        })
        # Projections for Deep Alignment (if needed)
        self.local_projections = nn.ModuleDict({
            m: ProjectionMLP(in_size=self.local_encoders[m].out_size, hidden=self.hidden_da)
            for m in self.modalities
        })
        self.cmc_loss_fn = MM_NTXent(self.batch_size, self.modalities, self.temperature)
        self.deep_alignment_loss_fn = DeepAlignmentLoss(modalities)

    def _forward_global(self, modality, inputs):
        x = inputs[modality]
        x, _ = self.global_encoders[modality](x)  # Global features
        x = nn.Flatten()(x)
        x_proj = self.global_projections[modality](x)
        return x_proj

    def _forward_local(self, modality, inputs):
        x = inputs[modality]
        _, local_features = self.local_encoders[modality](x)  # Local features
        return local_features

    def forward(self, x):
        global_outs = {}
        local_features = {}
        for m in self.modalities:
            global_outs[m] = self._forward_global(m, x)
            local_features[m] = self._forward_local(m, x)
        return global_outs, local_features

    def training_step(self, batch, batch_idx, optimizer_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        global_outs, local_features = self(batch)

        if optimizer_idx == 0:
            # Optimizer for CMC Loss
            loss_cmc, pos, neg = self.cmc_loss_fn(global_outs)
            self.log("cmc_loss", loss_cmc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("avg_positive_sim", pos, on_step=False, on_epoch=True, prog_bar=False)
            self.log("avg_neg_sim", neg, on_step=False, on_epoch=True, prog_bar=False)
            return loss_cmc

        elif optimizer_idx == 1:
            # Optimizer for Deep Alignment Loss
            loss_deep_alignment, loss_spatial, loss_temporal = self.deep_alignment_loss_fn(local_features)
            total_loss = self.beta * loss_deep_alignment
            self.log("deep_alignment_loss", loss_deep_alignment, on_step=False, on_epoch=True, prog_bar=True)
            self.log("loss_spatial", loss_spatial, on_step=False, on_epoch=True, prog_bar=False)
            self.log("loss_temporal", loss_temporal, on_step=False, on_epoch=True, prog_bar=False)
            loss_cmc = self.trainer.callback_metrics['cmc_loss']
            combined_loss = loss_cmc + total_loss
            self.log("total_loss", combined_loss, on_step=True, on_epoch=True, prog_bar=False)
            return total_loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        global_outs, local_features = self(batch)
        loss_cmc, _, _ = self.cmc_loss_fn(global_outs)
        loss_deep_alignment, _, _ = self.deep_alignment_loss_fn(local_features)
        total_loss = loss_cmc + self.beta * loss_deep_alignment
        self.log("ssl_val_loss", total_loss)

    def configure_optimizers(self):
        # Separate optimizers for CMC and Deep Alignment if needed
        optimizer_cmc = torch.optim.Adam(
            list(self.global_encoders.parameters()) + list(self.global_projections.parameters()), 
            lr=self.lr_cmc
        )
        optimizer_da = torch.optim.Adam(
            list(self.local_encoders.parameters()) + list(self.local_projections.parameters()), 
            lr=self.lr_da
        )
        # Combine optimizers
        return [optimizer_cmc, optimizer_da], []