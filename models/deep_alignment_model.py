# deep_alignment_model.py

import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from models.cmc_cvkm import ContrastiveMultiviewCodingCVKM

from models.cmc import ContrastiveMultiviewCoding
from torch.autograd import Variable

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
        N = self.batch_size

        # Computes cosine similarity matrix, shape: (N, N).
        # This computes the similarity between each sample in M1 with each sample in M2.
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

                # Add them to the positive set and exclude them from the negative set.
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
        for modality in features:
            intra_modality_similarities[modality] = self.similarity_metrics[modality].compute_similarity_matrix(batch, batch_idx)
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
        positive_indices = {}
        mask = torch.eye(self.batch_size, device=self.device)
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

        spatial_tokens_mod1, temporal_tokens_mod1 = local_features[self.modalities[0]]  # Inertial modality
        spatial_tokens_mod2, temporal_tokens_mod2 = local_features[self.modalities[1]]  # Skeleton modality

        # cost matrices for spatial tokens
        C_spatial = self.cost_matrix_batch_torch(spatial_tokens_mod1, spatial_tokens_mod2)
        # cost matrices for temporal tokens
        C_temporal = self.cost_matrix_batch_torch(temporal_tokens_mod1, temporal_tokens_mod2)

        bs, n_spatial, _ = spatial_tokens_mod1.size()
        m_spatial = spatial_tokens_mod2.size(1)
        bs_t, n_temporal, _ = temporal_tokens_mod1.size()
        m_temporal = temporal_tokens_mod2.size(1)

        miu_spatial = torch.ones(bs, n_spatial).to(C_spatial.device) / n_spatial  # batch_size x n_spatial
        nu_spatial = torch.ones(bs, m_spatial).to(C_spatial.device) / m_spatial   # batch_size x m_spatial
        miu_temporal = torch.ones(bs_t, n_temporal).to(C_temporal.device) / n_temporal
        nu_temporal = torch.ones(bs_t, m_temporal).to(C_temporal.device) / m_temporal

        # Wasserstein distances using IPOT algorithm
        spatial_loss = self.IPOT_distance_torch_batch(C_spatial, n_spatial, m_spatial, miu_spatial, nu_spatial)
        temporal_loss = self.IPOT_distance_torch_batch(C_temporal, n_temporal, m_temporal, miu_temporal, nu_temporal)

        # Total loss is the sum of spatial and temporal losses averaged over the batch
        loss = spatial_loss.mean() + temporal_loss.mean()
        return loss

    def cost_matrix_batch_torch(self, x, y):
        "Returns the cosine distance batchwise"
        # x: batch_size x n x D (embeddings from modality 1)
        # y: batch_size x m x D (embeddings from modality 2)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if y.dim() == 2:
            y = y.unsqueeze(1)
            
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")
   
                
        bs, n, D = x.size()
        m = y.size(1)
        x = x / (torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
        y = y / (torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(x, y.transpose(1, 2))  # batch_size x n x m
        cos_dis = 1 - cos_dis  # To minimize this value
        return cos_dis

    def IPOT_torch_batch(self, C, n, m, miu, nu, iteration=20):
        # C is the cost matrix: batch_size x n x m
        # miu and nu are the distributions over tokens in each modality: batch_size x n and batch_size x m
        bs = C.size(0)
        sigma = torch.ones(bs, m, 1).to(C.device) / m  # batch_size x m x 1
        T = torch.ones(bs, n, m).to(C.device)  # Transport plan
        C = torch.exp(-C / self.beta).float()
        for t in range(iteration):
            T = C * T  # batch_size x n x m
            for k in range(1):
                delta = miu.unsqueeze(2) / (T @ sigma + 1e-6)  # batch_size x n x 1
                sigma = nu.unsqueeze(2) / ((T.transpose(1, 2) @ delta) + 1e-6)  # batch_size x m x 1
            T = delta * T * sigma.transpose(1, 2)  # Update transport plan
        return T.detach()

    def IPOT_distance_torch_batch(self, C, n, m, miu, nu):
        # Computes the Wasserstein distance for each sample in the batch
        # C: batch_size x n x m
        bs = C.size(0)
        T = self.IPOT_torch_batch(C, n, m, miu, nu)
        distance = torch.sum(T * C, dim=(1, 2))  # batch_size
        return -distance  # Negative distance to minimize
    
class ContrastiveMultiviewCodingCVKMWithDeepAlignment(ContrastiveMultiviewCodingCVKM):
    """
    Extension to the CMC-CVKM to include local transformers and deep alignment loss for the local information.
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
        

    # def on_fit_start(self):
    #     for m in self.modalities:
    #         if m in self.similarity_metrics:
    #             self.similarity_metrics[m].move_to_device(self.device)
    #             self.similarity_metrics[m].set_dataset(self.trainer.datamodule.train_dataloader().dataset)


    #FABIAN override forward method to include deep alignment method
    
    def forward(self, x):
        adjusted_x = {}
        for m in self.modalities:
            if m == 'inertial':
                x_inertial = x[m]  
                # For the encoder, use as is
                x_inertial_encoder = x_inertial
                # For the local transformer (STDAT), keep as is
                #print(f"x_inertial shape before permutation: {x_inertial.shape}")
                x_inertial_local = x_inertial.permute(0, 2, 1)  # [64, 50, 6]
                #print(f"x_inertial shape after permutation: {x_inertial.shape}")
                adjusted_x[m] = {'encoder_input': x_inertial_encoder, 'local_transformer_input': x_inertial_local}
            elif m == 'skeleton':
                x_skeleton = x[m] # Shape: [64, 3, 50, 20]
                x_skeleton_encoder = x_skeleton
                # For the local transformer, permute to [B, F, J, C] = [64, 50, 20, 3]
                x_skeleton_local = x_skeleton.permute(0, 2, 3, 1)
                adjusted_x[m] = {'encoder_input': x_skeleton_encoder, 'local_transformer_input': x_skeleton_local}
            else:
                adjusted_x[m] = {'encoder_input': x[m], 'local_transformer_input': x[m]}
        
        h = {}
        for m in self.modalities:
            x_modality = adjusted_x[m]['encoder_input']
            #print(f"Global Encoder Input Shape ({m}): {x_modality.shape}")
            h[m] = self.encoders[m](x_modality)
            h[m] = h[m].reshape(h[m].size(0), -1)
            #print(f"Flattened Encoder Output Shape ({m}): {h[m].shape}")
        z = {}
        for m in self.modalities:
            z[m] = self.projections[m](h[m])
        global_features = z  
        
        local_features = {}
        for m in self.modalities:
            x_modality = adjusted_x[m]['local_transformer_input']
            #print(f"Local Transformer Input Shape ({m}): {x_modality.shape}")
            local_features[m] = self.local_transformers[m](x_modality)
        
        return global_features, local_features
    
    #FABIAN modified training step to include deep alignment loss
    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
            
        global_features, local_features = self(batch)
        
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
        loss_cmc_cmkm, _, _ = self.loss(global_features, batch, training = False, batch_idx = batch_idx)
        loss_deep_alignment = self.deep_alignment_loss_fn(local_features)
        total_loss = loss_cmc_cmkm + loss_deep_alignment
        
        self.log("ssl_val_loss", total_loss)