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
        beta=0.5,
        num_classes=27
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
        self.num_classes = num_classes

        self.save_hyperparameters(ignore=['global_encoders', 'local_encoders'])

        self.global_projections = nn.ModuleDict({
            m: ProjectionMLP(in_size=self.global_encoders[m].out_size, hidden=self.hidden_cmc)
            for m in self.modalities
        })
        self.local_projections = nn.ModuleDict({
            m: ProjectionMLP(in_size=self.local_encoders[m].out_size, hidden=self.hidden_da)
            for m in self.modalities
        })
        self.cmc_loss_fn = MM_NTXent(self.batch_size, self.modalities, self.temperature)
        self.deep_alignment_loss_fn = DeepAlignmentLoss(modalities)

        if isinstance(hidden_cmc, list) and len(hidden_cmc) > 0:
            final_global_dim = hidden_cmc[-1]
        else:
            final_global_dim = self.global_encoders[modalities[0]].out_size  # fallback

        total_dim_for_classifier = final_global_dim * len(modalities)
        self.classifier = nn.Linear(total_dim_for_classifier, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _forward_global(self, modality, inputs):
        x = inputs[modality]
        x, _ = self.global_encoders[modality](x)
        x = nn.Flatten()(x)
        x_proj = self.global_projections[modality](x)
        return x_proj

    def _forward_local(self, modality, inputs):
        x = inputs[modality]
        _, local_features = self.local_encoders[modality](x)
        return local_features

    def forward(self, x):
        global_outs = {}
        local_features = {}
        for m in self.modalities:
            global_outs[m] = self._forward_global(m, x)
            local_features[m] = self._forward_local(m, x)
        return global_outs, local_features

    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()

        global_outs, local_features = self(batch)

        loss_cmc, pos, neg = self.cmc_loss_fn(global_outs)
        loss_deep_alignment, loss_spatial, loss_temporal = self.deep_alignment_loss_fn(local_features)

        total_loss = loss_cmc + self.beta * loss_deep_alignment

        if 'label' in batch:
            concatenated_feats = torch.cat(list(global_outs.values()), dim=1)
            logits = self.classifier(concatenated_feats)
            class_loss = self.criterion(logits, batch['label'].long() - 1)
            total_loss += class_loss
        
        self.log("cmc_loss", loss_cmc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("deep_alignment_loss", loss_deep_alignment, on_step=False, on_epoch=True, prog_bar=False)
        self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("avg_positive_sim", pos, on_step=False, on_epoch=True, prog_bar=False)
        self.log("avg_neg_sim", neg, on_step=False, on_epoch=True, prog_bar=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        global_outs, local_features = self(batch)
        loss_cmc, _, _ = self.cmc_loss_fn(global_outs)
        loss_deep_alignment, _, _ = self.deep_alignment_loss_fn(local_features)
        total_loss = loss_cmc + self.beta * loss_deep_alignment

        logits = None
        preds = None
        if 'label' in batch:
            concatenated_feats = torch.cat(list(global_outs.values()), dim=1)
            logits = self.classifier(concatenated_feats)
            class_loss = self.criterion(logits, batch['label'].long() - 1)
            total_loss += class_loss

            preds = torch.argmax(logits, dim=1)

        self.log("total_val_loss", total_loss, prog_bar=True)
        self.log("val_cmc_loss", loss_cmc)
        self.log("val_da_loss", loss_deep_alignment)

        out = {"val_loss": total_loss}
        if logits is not None and preds is not None:
            out["logits"] = logits
            out["labels"] = batch["label"] - 1
            out["preds"] = preds
        return out

    def configure_optimizers(self):
        all_params = (
            list(self.global_encoders.parameters())
            + list(self.global_projections.parameters())
            + list(self.local_encoders.parameters())
            + list(self.local_projections.parameters())
            + list(self.classifier.parameters())
        )
        return torch.optim.Adam(all_params, lr=self.lr_cmc)