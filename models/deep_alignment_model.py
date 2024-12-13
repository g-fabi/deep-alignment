# deep_alignment_model.py

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from models.cmc_cvkm import ContrastiveMultiviewCodingCVKM

from models.cmc import ContrastiveMultiviewCoding
from torch.autograd import Variable

from utils.training_utils import init_encoders, init_local_transformer

from models.cmc_cvkm import MM_NTXent_CVKM

class MM_NTXent_CVKM(LightningModule):
    """
    Multimodal adaptation of NTXent, which uses cross-view knowledge mining
    (based on the given similarity metrics) to guide the training process.

    NOTE: currently assumes that there are exactly 2 modalities.
    """
    def __init__(self, batch_size, modalities, similarity_metrics, cmkm_config, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = modalities
        self.similarity_metrics = similarity_metrics
        self.cmkm_config = cmkm_config
        self.temperature = temperature

    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2):
        """Returns an [N, N] matrix of cosine similarities."""
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        similarity_matrix = torch.matmul(features_1, features_2.T)
        return similarity_matrix

    def forward(self, features, batch, training, batch_idx):
        # Let M1 and M2 be abbreviations for the first and the second modality, respectively.

        # Computes cosine similarity matrix, shape: (N, N).
        # This computes the similarity between each sample in M1 with each sample in M2.
        N = features[self.modalities[0]].size(0)
        features_1 = features[self.modalities[0]]
        features_2 = features[self.modalities[1]]
        similarity_matrix = MM_NTXent_CVKM.get_cosine_sim_matrix(features_1, features_2)

        # We need to formulate (2 * N) instance discrimination problems:
        # -> each instance from M1 with each instance from M2
        # -> each instance from M2 with each instance from M1
        # The first set of rows is similarity from M1 to M2; the second set of rows is similarity from M2 to M1.
        inter_modality_similarities = torch.cat([similarity_matrix, similarity_matrix.T], dim=0)

        # Similarities of positive pairs are on the main diagonal for each submatrix.
        # The rest are similarities for negative pairs.
        positives_mask = torch.eye(N).bool().repeat([2, 1]).to(self.device)
        negatives_mask = ~positives_mask
        print("[Debug] positives_mask:", positives_mask.shape, positives_mask.device, positives_mask.dtype)
        # TODO maybe: build a big matrix with all 4 similarities, like the diagram in the paper? maybe it would simplify some of this logic.

        # Cross-view knowledge mining, only applied during training and depending on the provided cmkm_config.
        if training:
            # Use the provided similarity metric to compute the intra-modality similarities and connectivity.
            if self.cmkm_config['positive_mining_enabled'] or self.cmkm_config['negative_set_pruning_enabled'] or self.cmkm_config["loss_term_weighing_enabled"]:
                intra_modality_similarities = self.compute_intra_modality_similarities(features, batch, batch_idx)
                connectivity = self.compute_connectivity(intra_modality_similarities)

            # Positive pair mining
            if self.cmkm_config['positive_mining_enabled'] == True:
                positive_indices = self.mine_positives(intra_modality_similarities, K=self.cmkm_config['positive_mining_k'])
                if self.cmkm_config['positive_mining_symmetric'] == True:
                    # Symmetric means we consider as positives for both modalities all samples returned by the positive mining procedure.
                    merged_indices = torch.cat([positive_indices[self.modalities[0]], positive_indices[self.modalities[1]]], dim=1)
                    positive_indices = {
                        self.modalities[0]: merged_indices,
                        self.modalities[1]: merged_indices
                    }

                # Debug: Check shapes, devices, dtypes
                print("[Debug] positive_indices[mod1]:", positive_indices[self.modalities[1]].shape,
                      positive_indices[self.modalities[1]].device, positive_indices[self.modalities[1]].dtype)
                print("[Debug] torch.arange(N):", torch.arange(N).shape, torch.arange(N).device, torch.arange(N).dtype)
                
                # Ensure that positive_indices are on the same device as positives_mask
                positive_indices[self.modalities[0]] = positive_indices[self.modalities[0]].to(positives_mask.device)
                positive_indices[self.modalities[1]] = positive_indices[self.modalities[1]].to(positives_mask.device)

                positives_mask[torch.arange(N, device=positives_mask.device).unsqueeze(1), positive_indices[self.modalities[1]]] = True
                negatives_mask[torch.arange(N, device=positives_mask.device).unsqueeze(1), positive_indices[self.modalities[1]]] = False
                positives_mask[torch.arange(N, 2*N, device=positives_mask.device).unsqueeze(1), positive_indices[self.modalities[0]]] = True
                negatives_mask[torch.arange(N, 2*N, device=positives_mask.device).unsqueeze(1), positive_indices[self.modalities[0]]] = False

                positives_mask[torch.arange(N).unsqueeze(1), positive_indices[self.modalities[1]]] = True
                negatives_mask[torch.arange(N).unsqueeze(1), positive_indices[self.modalities[1]]] = False
                positives_mask[torch.arange(N, 2*N).unsqueeze(1), positive_indices[self.modalities[0]]] = True
                negatives_mask[torch.arange(N, 2*N).unsqueeze(1), positive_indices[self.modalities[0]]] = False

            # Negative set pruning.
            if self.cmkm_config['negative_set_pruning_enabled'] == True:
                gamma = self.cmkm_config['negative_set_pruning_threshold']
                
                pruned_negative_indices = {
                    self.modalities[0]: connectivity[self.modalities[0]] > gamma,
                    self.modalities[1]: connectivity[self.modalities[1]] > gamma
                }
                # Debug: Check pruned_negative_indices
                print("[Debug] pruned_negative_indices[mod1]:", pruned_negative_indices[self.modalities[1]].shape,
                      pruned_negative_indices[self.modalities[1]].device, pruned_negative_indices[self.modalities[1]].dtype)

                # Exclude them from the negative sets of each modality.
                negatives_mask[torch.arange(N).unsqueeze(1), pruned_negative_indices[self.modalities[1]]] = False
                negatives_mask[torch.arange(N, 2*N).unsqueeze(1), pruned_negative_indices[self.modalities[0]]] = False

        # Compute the mean positive and negative similarities, for logging purposes.
        mean_positive_similarities = torch.mean(inter_modality_similarities[positives_mask])
        mean_negative_similarities = torch.mean(inter_modality_similarities[negatives_mask])

        # Apply temperature scaling.
        inter_modality_similarities = inter_modality_similarities / self.temperature

        # Compute NTXEnt loss.
        exp_similarities = torch.exp(inter_modality_similarities)
        numerators = torch.sum(exp_similarities * positives_mask, dim=1)
        denominators = torch.sum(exp_similarities * negatives_mask, dim=1)

        # Compute this here so we only do it once.
        if training and (self.cmkm_config["intra_modality_negatives_enabled"] or self.cmkm_config["positive_mining_enabled"]):
            intra_modality_similarities_1 = MM_NTXent_CVKM.get_cosine_sim_matrix(features_1, features_1)
            intra_modality_similarities_2 = MM_NTXent_CVKM.get_cosine_sim_matrix(features_2, features_2)
            intra_modality_exp_similarities = torch.exp(torch.cat([intra_modality_similarities_1, intra_modality_similarities_2], dim=0))

        # Add intra-modality negatives to preserve similarities in the latent space of each modality.
        if training and self.cmkm_config["intra_modality_negatives_enabled"] == True:
            intra_modality_negatives = ~(torch.eye(N).bool()).repeat([2, 1]).to(self.device)

            # If also doing positive mining, then adjust the intra-modality negative set as well by excluding the mined positive samples.
            # They will already appear in the denominator of the loss, when the numerator is added, so we don't want to count them twice.
            if self.cmkm_config['positive_mining_enabled'] == True:
                intra_modality_negatives[torch.arange(N).unsqueeze(1), positive_indices[self.modalities[0]]] = False
                intra_modality_negatives[torch.arange(N, 2*N).unsqueeze(1), positive_indices[self.modalities[1]]] = False

            # If also doing negative set pruning, then adjust the intra-modality negative set as well by excluding the pruned samples.
            if self.cmkm_config['negative_set_pruning_enabled'] == True:
                intra_modality_negatives[torch.arange(N).unsqueeze(1), pruned_negative_indices[self.modalities[0]]] = False
                intra_modality_negatives[torch.arange(N, 2*N).unsqueeze(1), pruned_negative_indices[self.modalities[1]]] = False

            lambda_ = self.cmkm_config["intra_modality_negatives_weight"]
            denominators += lambda_ * torch.sum(intra_modality_exp_similarities * intra_modality_negatives, dim=1)

        # If doing positive mining AND using intra-modality negatives, then also add the intra-modality positives.
        if training and self.cmkm_config["positive_mining_enabled"] and self.cmkm_config["intra_modality_negatives_enabled"]:
            intra_modality_positives = torch.zeros([2*N, N]).bool().to(self.device)
            intra_modality_positives[torch.arange(N).unsqueeze(1), positive_indices[self.modalities[0]]] = True
            intra_modality_positives[torch.arange(N, 2*N).unsqueeze(1), positive_indices[self.modalities[1]]] = True
            numerators += torch.sum(intra_modality_exp_similarities * intra_modality_positives, dim=1)

        losses_per_sample = -torch.log((numerators / (numerators + denominators)))
        if training:
            if self.cmkm_config["loss_term_weighing_enabled"] == True:
                weights = self.compute_loss_term_weights(connectivity, scale=self.cmkm_config["loss_term_weighing_scale"])
                losses_per_sample = losses_per_sample * weights
        final_loss = torch.mean(losses_per_sample)

        return final_loss, mean_positive_similarities, mean_negative_similarities
    

    def compute_intra_modality_similarities(self, features, batch, batch_idx):
        """For each modality, returns an [N, N]-shaped tensor of intramodality similarities using the provided similarity metric."""
        intra_modality_similarities = {}
        if self.similarity_metrics:
            for modality in features:
                intra_modality_similarities[modality] = \
                    self.similarity_metrics[modality].compute_similarity_matrix(batch, batch_idx)
        else:
            for modality in features:
                feats = features[modality]
                # Normalize the features
                feats = F.normalize(feats, dim=1)
                # Compute cosine similarity matrix
                sim_matrix = torch.matmul(feats, feats.T)
                intra_modality_similarities[modality] = sim_matrix
        return intra_modality_similarities

    def compute_connectivity(self, intra_modality_similarities):
        """For each modality, return an N-shaped tensor of connectivity values using the provided similarity matrix."""
        connectivity = {}
        for modality in self.modalities:
            connectivity[modality] = torch.mean(intra_modality_similarities[modality], dim=0)
        return connectivity

    def mine_positives(self, intra_modality_similarities, K=1):
        """
        Uses the provided similarity metric to find potential positives based on intra-modal similarity.
        For each modality, returns an [N, K]-shaped tensor of indices.
        """
        N = intra_modality_similarities[self.modalities[0]].size(0)
        positive_indices = {}
        mask = torch.eye(N, device=self.device)
        for modality in self.modalities:
            # Mask out the main diagonal (which are all equal to 1) and compute the topK highest similarities.
            topk_indices = torch.topk(intra_modality_similarities[modality] - mask, k=K).indices

            positive_indices[modality] = topk_indices
        return positive_indices

    def compute_loss_term_weights(self, connectivity, scale=0.05):
        weights = torch.cat([connectivity[self.modalities[0]], connectivity[self.modalities[1]]])
        weights = torch.exp(weights * scale)
        return weights


#FABIAN: WD loss
#https://github.com/LiqunChen0606/Graph-Optimal-Transport/blob/master/BAN_vqa/OT_torch_.py
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

        print(f"spatial_tokens_mod1 shape: {spatial_tokens_mod1.shape}")
        print(f"spatial_tokens_mod2 shape: {spatial_tokens_mod2.shape}")
        print(f"temporal_tokens_mod1 shape: {temporal_tokens_mod1.shape}")
        print(f"temporal_tokens_mod2 shape: {temporal_tokens_mod2.shape}")

        # Collapse seq_len and n_joints/n_frames
        bs, seq_len, n_joints, dim = spatial_tokens_mod2.shape
        spatial_tokens_mod2 = spatial_tokens_mod2.view(bs, seq_len * n_joints, dim)
        n_spatial_mod2 = spatial_tokens_mod2.size(1)

        bs, seq_len, n_frames, dim = temporal_tokens_mod2.shape
        temporal_tokens_mod2 = temporal_tokens_mod2.view(bs, seq_len * n_frames, dim)
        n_temporal_mod2 = temporal_tokens_mod2.size(1)

        # Verify that spatial_tokens_mod1 and spatial_tokens_mod2 now have matching dimensions
        print(f"spatial_tokens_mod1 shape: {spatial_tokens_mod1.shape}")  # [bs, n_spatial, dim]
        print(f"spatial_tokens_mod2 shape: {spatial_tokens_mod2.shape}")  # [bs, n_spatial_mod2, dim]

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
        loss = loss_spatial + loss_temporal
        return loss

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
        return -cost.mean()

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

class CustomMM_NTXent_CVKM(MM_NTXent_CVKM):
    def forward(self, features, batch, training, batch_idx):
        # Derive N from actual batch size
        N = features[self.modalities[0]].size(0)
        device = features[self.modalities[0]].device
        print(f"[Debug] Batch size (N): {N}")
        print(f"[Debug] Device: {device}")

        features_1 = features[self.modalities[0]]
        features_2 = features[self.modalities[1]]
        similarity_matrix = self.get_cosine_sim_matrix(features_1, features_2)
        print(f"[Debug] similarity_matrix shape: {similarity_matrix.shape}")

        inter_modality_similarities = torch.cat([similarity_matrix, similarity_matrix.T], dim=0)
        print(f"[Debug] inter_modality_similarities shape: {inter_modality_similarities.shape}")

        positives_mask = torch.eye(N).bool().repeat(2, 1).to(device)
        negatives_mask = ~positives_mask
        print(f"[Debug] positives_mask shape: {positives_mask.shape}, device: {positives_mask.device}, dtype: {positives_mask.dtype}")
        print(f"[Debug] negatives_mask shape: {negatives_mask.shape}, device: {negatives_mask.device}, dtype: {negatives_mask.dtype}")

        if training:
            cmkm_config = self.cmkm_config
            if (cmkm_config.get('positive_mining_enabled', False) or
                cmkm_config.get('negative_set_pruning_enabled', False) or
                cmkm_config.get('loss_term_weighing_enabled', False)):
                intra_modality_similarities = self.compute_intra_modality_similarities(features, batch, batch_idx)
                connectivity = self.compute_connectivity(intra_modality_similarities)

            if cmkm_config.get('positive_mining_enabled', False):
                positive_mining_k = cmkm_config.get('positive_mining_k', 1)
                positive_indices = self.mine_positives(intra_modality_similarities, K=positive_mining_k)

                if cmkm_config.get('positive_mining_symmetric', False):
                    merged_indices = torch.cat([positive_indices[self.modalities[0]], positive_indices[self.modalities[1]]], dim=1)
                    positive_indices = {
                        self.modalities[0]: merged_indices,
                        self.modalities[1]: merged_indices
                    }

                print(f"[Debug] positive_indices[{self.modalities[1]}] shape:", positive_indices[self.modalities[1]].shape,
                      positive_indices[self.modalities[1]].device, positive_indices[self.modalities[1]].dtype)
                print(f"[Debug] positive_indices[{self.modalities[1]}]: {positive_indices[self.modalities[1]]}")
                print(f"[Debug] torch.arange(N).unsqueeze(1).shape:", torch.arange(N).unsqueeze(1).shape,
                      torch.arange(N).unsqueeze(1).device, torch.arange(N).unsqueeze(1).dtype)
                print(f"[Debug] positives_mask.shape:", positives_mask.shape)

                # Ensure same device and dtype
                positive_indices[self.modalities[0]] = positive_indices[self.modalities[0]].to(device)
                positive_indices[self.modalities[1]] = positive_indices[self.modalities[1]].to(device)

                try:
                    positives_mask[torch.arange(N, device=device), positive_indices[self.modalities[1]]] = True
                    negatives_mask[torch.arange(N, device=device), positive_indices[self.modalities[1]]] = False
                    positives_mask[torch.arange(N, 2*N, device=device), positive_indices[self.modalities[0]]] = True
                    negatives_mask[torch.arange(N, 2*N, device=device), positive_indices[self.modalities[0]]] = False
                except Exception as e:
                    print(f"[Error] Exception during positive mining indexing: {e}")
                    print(f"[Debug] positives_mask size: {positives_mask.size()}")
                    print(f"[Debug] negatives_mask size: {negatives_mask.size()}")
                    print(f"[Debug] positive_indices[{self.modalities[1]}] size: {positive_indices[self.modalities[1]].size()}")
                    print(f"[Debug] positive_indices[{self.modalities[0]}] size: {positive_indices[self.modalities[0]].size()}")
                    raise e

            if cmkm_config.get('negative_set_pruning_enabled', False):
                gamma = cmkm_config.get('negative_set_pruning_threshold', 0.5)
                pruned_negative_indices = {
                    self.modalities[0]: connectivity[self.modalities[0]] > gamma,
                    self.modalities[1]: connectivity[self.modalities[1]] > gamma
                }

                print(f"[Debug] pruned_negative_indices[{self.modalities[1]}] shape:", pruned_negative_indices[self.modalities[1]].shape,
                      pruned_negative_indices[self.modalities[1]].device, pruned_negative_indices[self.modalities[1]].dtype)
                print(f"[Debug] pruned_negative_indices[{self.modalities[1]}]:", pruned_negative_indices[self.modalities[1]])

                # Ensure same device
                pruned_negative_indices[self.modalities[0]] = pruned_negative_indices[self.modalities[0]].to(device)
                pruned_negative_indices[self.modalities[1]] = pruned_negative_indices[self.modalities[1]].to(device)

                try:
                    negatives_mask[torch.arange(N, device=device).unsqueeze(1), pruned_negative_indices[self.modalities[1]]] = False
                    negatives_mask[torch.arange(N, 2*N, device=device).unsqueeze(1), pruned_negative_indices[self.modalities[0]]] = False
                except Exception as e:
                    print(f"[Error] Exception during negative set pruning indexing: {e}")
                    raise e

        mean_positive_similarities = torch.mean(inter_modality_similarities[positives_mask])
        mean_negative_similarities = torch.mean(inter_modality_similarities[negatives_mask])
        print(f"[Debug] mean_positive_similarities: {mean_positive_similarities.item()}")
        print(f"[Debug] mean_negative_similarities: {mean_negative_similarities.item()}")

        inter_modality_similarities = inter_modality_similarities / self.temperature
        exp_similarities = torch.exp(inter_modality_similarities)
        numerators = torch.sum(exp_similarities * positives_mask, dim=1)
        denominators = torch.sum(exp_similarities * negatives_mask, dim=1)

        if training and (self.cmkm_config.get('intra_modality_negatives_enabled', False) or
                         self.cmkm_config.get('positive_mining_enabled', False)):
            intra_modality_similarities_1 = self.get_cosine_sim_matrix(features_1, features_1)
            intra_modality_similarities_2 = self.get_cosine_sim_matrix(features_2, features_2)
            intra_modality_exp_similarities = torch.exp(torch.cat([intra_modality_similarities_1, intra_modality_similarities_2], dim=0)).to(device)

        if training and self.cmkm_config.get('intra_modality_negatives_enabled', False):
            intra_modality_negatives = ~(torch.eye(N).bool()).repeat(2, 1).to(device)

            if self.cmkm_config.get('positive_mining_enabled', False):
                intra_modality_negatives[torch.arange(N, device=device).unsqueeze(1), positive_indices[self.modalities[0]]] = False
                intra_modality_negatives[torch.arange(N, 2*N, device=device).unsqueeze(1), positive_indices[self.modalities[1]]] = False

            if self.cmkm_config.get('negative_set_pruning_enabled', False):
                intra_modality_negatives[torch.arange(N, device=device).unsqueeze(1), pruned_negative_indices[self.modalities[0]]] = False
                intra_modality_negatives[torch.arange(N, 2*N, device=device).unsqueeze(1), pruned_negative_indices[self.modalities[1]]] = False

            lambda_ = self.cmkm_config.get('intra_modality_negatives_weight', 1.0)
            denominators += lambda_ * torch.sum(intra_modality_exp_similarities * intra_modality_negatives, dim=1)

        if training and self.cmkm_config.get('positive_mining_enabled', False) and self.cmkm_config.get('intra_modality_negatives_enabled', False):
            intra_modality_positives = torch.zeros(2*N, N, dtype=torch.bool, device=device)
            intra_modality_positives[torch.arange(N, device=device).unsqueeze(1), positive_indices[self.modalities[0]]] = True
            intra_modality_positives[torch.arange(N, 2*N, device=device).unsqueeze(1), positive_indices[self.modalities[1]]] = True
            numerators += torch.sum(intra_modality_exp_similarities * intra_modality_positives, dim=1)

        losses_per_sample = -torch.log(numerators / (numerators + denominators))

        if training and self.cmkm_config.get('loss_term_weighing_enabled', False):
            weights = self.compute_loss_term_weights(connectivity, scale=self.cmkm_config.get('loss_term_weighing_scale', 0.05))
            losses_per_sample = losses_per_sample * weights

        final_loss = torch.mean(losses_per_sample)
        return final_loss, mean_positive_similarities, mean_negative_similarities
    
class ContrastiveMultiviewCodingCVKMWithDeepAlignment(ContrastiveMultiviewCodingCVKM):
    """
    Extension to the CMC-CVKM to include encoders that handle both global and local representations.
    """
    def __init__(
        self,
        modalities,
        encoders,
        local_transformers,
        similarity_metrics,
        cmkm_config,
        hidden=[256, 128],
        batch_size=64,
        temperature=0.1,
        optimizer_name_ssl='adam',
        lr=0.001,
        **kwargs
    ):
        # Initialize traditional CMC.
        super().__init__(
            modalities=modalities,
            encoders=encoders,
            similarity_metrics=similarity_metrics,
            cmkm_config=cmkm_config,                         
            hidden=hidden,
            batch_size=batch_size,
            temperature=temperature,
            optimizer_name_ssl=optimizer_name_ssl,
            lr=lr,
            **kwargs)

        self.local_transformers = nn.ModuleDict(local_transformers)

        self.deep_alignment_loss_fn = DeepAlignmentLoss(modalities)
        
        self.loss = CustomMM_NTXent_CVKM(
            batch_size=batch_size,
            modalities=modalities,
            similarity_metrics=similarity_metrics,
            cmkm_config=cmkm_config,
            temperature=temperature
        )
        
    def on_fit_start(self):
        for m in self.similarity_metrics:
            self.similarity_metrics[m].move_to_device(self.device)

    def forward(self, batch):
        for m in self.modalities:
            batch[m] = batch[m].to(self.device)

        global_features = {}
        local_features = {}

        for m in self.modalities:
            # Use the global encoder
            global_feat, _ = self.encoders[m](batch[m])
            global_features[m] = global_feat

            # Use the local transformer
            _, local_feat = self.local_transformers[m](batch[m])
            local_features[m] = local_feat

        # Apply projection for contrastive learning if needed
        z = {}
        for m in self.modalities:
            z[m] = self.projections[m](global_features[m])
        global_features = z

        return global_features, local_features
    
    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        
        global_features, local_features = self(batch)
        
        # Include global features in the batch for RawSimilarityMetric
        batch['global_features'] = global_features
        
        # Debugging: Check shapes
        N = batch[self.modalities[0]].size(0)
        print(f"\n[Debug] Batch size (N): {N}")
        for m in self.modalities:
            print(f"[Debug] Global features [{m}] shape: {global_features[m].shape}, device: {global_features[m].device}, dtype: {global_features[m].dtype}")
            print(f"[Debug] Batch data [{m}] shape: {batch[m].shape}, device: {batch[m].device}, dtype: {batch[m].dtype}")

        loss_cmc_cmkm, pos, neg = self.loss(global_features, batch, training=True, batch_idx=batch_idx)
        loss_deep_alignment = self.deep_alignment_loss_fn(local_features)
        total_loss = loss_cmc_cmkm + loss_deep_alignment
            
        self.log("ssl_train_loss", total_loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        self.log("deep_alignment_loss", loss_deep_alignment)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        
        global_features, local_features = self(batch)
        batch['global_features'] = global_features

        loss_cmc_cmkm, _, _ = self.loss(global_features, batch, training=False, batch_idx=batch_idx)
        loss_deep_alignment = self.deep_alignment_loss_fn(local_features)
        total_loss = loss_cmc_cmkm + loss_deep_alignment
        
        self.log("ssl_val_loss", total_loss)