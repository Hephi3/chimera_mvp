import torch
import torch.nn as nn
import torch.nn.functional as F
from models.clam_models.model_clam import CLAM_SB, Attn_Net_Gated
from models.cd_models.cd_nn import CD_Solo

class MultimodalFusionSimultaneousModel(nn.Module):
    def __init__(self, gate=True, size_arg="tiny", dropout=0.25, 
             k_sample=8, n_classes=3, instance_loss_fn=None, 
             subtyping=False, clinical_dim=32, embed_dim=1024, 
             norm=False, attention_supervision_weight=0.0,
             **kwargs):
        
        self.kind = 'multimodal'
        super(MultimodalFusionSimultaneousModel, self).__init__()
        
        print("Initializing MultimodalFusionSimultaneousModel with attention supervision weight:", attention_supervision_weight)
        
        # Initialize CLAM_SB model with unpacked arguments
        self.clam = CLAM_SB(gate=gate, size_arg=size_arg, dropout=dropout,
                       k_sample=k_sample, n_classes=n_classes,
                       instance_loss_fn=instance_loss_fn,
                       subtyping=subtyping, embed_dim=embed_dim, 
                       attention_supervision_weight = attention_supervision_weight
                       )

        size_arg = kwargs.get('size', 'tiny')  # Default to 'tiny' if not provided
        dropout = kwargs.get('dropout', 0.5)  # Default dropout value
        self.norm = norm
        
        # Get the feature dimension from CLAM's size_dict
        self.clam_features_dim = self.clam.size_dict[size_arg][1]  # 512 for both small and big
        
        # Initialize clinical data model
        self.clinical_model = CD_Solo(input_dim=23, dropout=dropout, return_features=True) 
        self.clinical_features_dim = clinical_dim  # Based on CD_Solo_Small architecture
        
        if self.norm:
            self.clam_norm = nn.LayerNorm(self.clam_features_dim)
            self.clinical_norm = nn.LayerNorm(self.clinical_features_dim)
            self.clam_weight = nn.Parameter(torch.ones(1))
            self.clinical_weight = nn.Parameter(torch.ones(1))
        
        # Learnable loss weights (log-scale preferred for stability)
        self.loss_weight_mm = nn.Parameter(torch.tensor(0.0))  # log(1)
        self.loss_weight_clam = nn.Parameter(torch.tensor(0.0))
        self.loss_weight_cd = nn.Parameter(torch.tensor(0.0))
        
        # Feature fusion
        self.fusion_dim = self.clam_features_dim + self.clinical_features_dim
        # print(f"Dimensions: CLAM: {self.clam_features_dim}, Clinical: {self.clinical_features_dim}, Fusion: {self.fusion_dim}")
        self.fusion_net = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Final classifier
        self.classifier = nn.Linear(128, n_classes)
        
    def fusion(self, h_path, h_clinical):
        """
        Concatenate image and clinical features.
        """
        # h_clinical = h_clinical.unsqueeze(0)
        h_combined = torch.cat([h_path, h_clinical], dim=1)
        h_fused = self.fusion_net(h_combined)
        return h_fused
    
    def freeze_fusion(self):
        """
        Freeze the fusion network parameters.
        """
        for param in self.fusion_net.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("Fusion network parameters are frozen.")
    
    def unfreeze_fusion(self):
        """
        Unfreeze the fusion network parameters.
        """
        for param in self.fusion_net.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Fusion network parameters are unfrozen.")
    
    def forward(self, h, clinical_features, label=None, instance_eval=False, return_features=False, tumor_labels=None):
        """
        h: Image features from dataloader
        clinical_features: Clinical data features
        """
        
        # Process WSI data through CLAM (gets features instead of logits due to Identity)
        logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam = self.clam(h, label, instance_eval, return_features=True, tumor_labels=tumor_labels)
        
        # Get image features from CLAM results
        h_path = results_clam['features']  # This is the M matrix (aggregated features)
        
        if tumor_labels is not None:
            attention_supervision_loss = results_clam['attention_supervision_loss']

        # Process clinical data - output is features due to Identity
        if self.clinical_features_dim == 23:
            # If clinical features are already in the correct format
            h_clinical = clinical_features # Use clinical_features directly
            logits_clinical = None  # No logits for clinical features
            Y_prob_clinical = None
            Y_hat_clinical = None
            
        else:
            logits_clinical, h_clinical = self.clinical_model(clinical_features) 
            
            # Add 1 dimension for batch processing
            # logits_clinical = logits_clinical.unsqueeze(0)  # Ensure clinical features
            
            Y_prob_clinical = F.softmax(logits_clinical, dim=1)
            Y_hat_clinical = torch.argmax(Y_prob_clinical, dim=1)
    
        if self.norm:
            h_path = self.clam_norm(h_path)
            h_clinical = self.clinical_norm(h_clinical)
        
            h_path = h_path * F.softplus(self.clam_weight)
            h_clinical = h_clinical * F.softplus(self.clinical_weight)
        
        h_fused = self.fusion(h_path, h_clinical)
        
        # Final classification
        logits = self.classifier(h_fused)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)
            
        return {"CLAM": [logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam], "CD": [logits_clinical, Y_prob_clinical, Y_hat_clinical, None, None], "MM": [logits, Y_prob, Y_hat, None, None], "attention_supervision_loss": attention_supervision_loss if tumor_labels is not None else None}

class MultimodalAuxiliarySupervisionModel(MultimodalFusionSimultaneousModel):
    def __init__(self, *args, **kwargs):
        super(MultimodalAuxiliarySupervisionModel, self).__init__(*args, **kwargs)
        self.clinical_predictor = nn.Sequential(
            nn.Linear(self.clam_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 23),
        )

    def forward(self, h, clinical_features, label=None, instance_eval=False, return_features=False):
        result_dict = super(MultimodalAuxiliarySupervisionModel, self).forward(h, clinical_features, label, instance_eval, return_features)
        clam_features = result_dict['CLAM'][4]['features']
        
        clinical_pred = self.clinical_predictor(clam_features)
        clinical_pred_loss = F.mse_loss(clinical_pred, clinical_features.unsqueeze(0))
        result_dict['Clinical_Prediction'] = clinical_pred_loss
        return result_dict
        
        

















class MultimodalGatedModel(MultimodalFusionSimultaneousModel):
    def __init__(self, *args, **kwargs):
        super(MultimodalGatedModel, self).__init__(*args, **kwargs)

        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def fusion(self, h_path, h_clinical):        
        h_clinical = h_clinical.unsqueeze(0)
        h_combined = torch.cat([h_path, h_clinical], dim=1)
        weights = self.gate(h_combined)
        w_clam, w_clinical = weights[:, 0:1], weights[:, 1:2]
        fused = w_clam * h_path + w_clinical * h_clinical
        return self.fusion_net(fused)
        



class MultimodalFuSimBalancedAttentionModel(MultimodalFusionSimultaneousModel):
    def __init__(self, *args, **kwargs):
        super(MultimodalFuSimBalancedAttentionModel, self).__init__(*args, **kwargs)
        
        self.path_projection = nn.Linear(self.clam_features_dim, 128)
        self.clinical_projection = nn.Linear(self.clinical_features_dim, 128)
        
        self.attention = Attn_Net_Gated(
            L=128,
            D=64,
            dropout=True,
            n_classes=1
        )
        self.post_attention = nn.Sequential(
            nn.Linear(128 * 2, 128),  # Combine both modalities
            nn.ReLU(),
            nn.Dropout(0.5)
        )
            
    
    def fusion(self, h_path, h_clinical):
        """
        Fuse image and clinical features using attention.
        """
        # Ensure h_clinical has batch dimension
        if h_clinical.dim() == 1:
            h_clinical = h_clinical.unsqueeze(0)
            
        # Project features to same dimension
        h_path_proj = self.path_projection(h_path)
        h_clinical_proj = self.clinical_projection(h_clinical)
        
        # Stack features for attention (along sequence dimension)
        # Shape: [batch_size, 2, 128]
        features = torch.stack([h_path_proj, h_clinical_proj], dim=1)
        
        # Apply attention to get weights
        batch_size = features.size(0)
        features_reshaped = features.view(batch_size * 2, 128)  # Reshape for attention
        
        A, features_reshaped = self.attention(features_reshaped)
        A = torch.transpose(A, 1, 0)  # Make it [1, batch_size*2]
        A = F.softmax(A, dim=1)  # Softmax over features
        
        # Reshape attention weights back
        A = A.view(1, batch_size, 2)  # [1, batch_size, 2]
        
        # Apply attention weights
        features_weighted = torch.bmm(A, features)  # [batch_size, 1, 128]
        features_weighted = features_weighted.squeeze(1)  # [batch_size, 128]
        
        # Concatenate weighted features with original features for residual connection
        h_combined = torch.cat([features_weighted, h_path_proj], dim=1)  # [batch_size, 256]
        
        # Final processing
        h_fused = self.post_attention(h_combined)
        
        return h_fused


class MultimodalFusionContrastiveModel(MultimodalFusionSimultaneousModel):
    def __init__(self, contrastive_dim=128, use_laaf=True, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_dim = contrastive_dim
        self.use_laaf = use_laaf
        self.temperature = temperature

        # Projection heads for contrastive learning
        self.proj_img = nn.Sequential(
            nn.Linear(self.clam_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, contrastive_dim)
        )
        self.proj_tab = nn.Sequential(
            nn.Linear(self.clinical_features_dim + (1 if use_laaf else 0), 256),
            nn.ReLU(),
            nn.Linear(256, contrastive_dim)
        )

    def contrastive_loss(self, z1, z2):
        """
        Enhanced contrastive loss that works with any batch size.
        """
        # For single sample case, create synthetic negatives
        if z1.size(0) == 1:
            # Create synthetic negative by permuting features
            neg_z2 = z2.roll(shifts=1, dims=1)  # Shift features to create a negative
            
            # Compute positive and negative similarities
            pos_sim = F.cosine_similarity(z1, z2)
            neg_sim = F.cosine_similarity(z1, neg_z2)
            
            # Simple InfoNCE-style loss
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim.unsqueeze(0)], dim=0)
            logits = logits / self.temperature
            labels = torch.zeros(1, dtype=torch.long).to(z1.device)  # First is positive
            # [2,1] to [1,2] for cross-entropy
            logits = logits.view(1, -1)
            
            return F.cross_entropy(logits, labels)
        
        # Normal implementation for batch size > 1
        else:
            logits = torch.matmul(z1, z2.T) / self.temperature
            labels = torch.arange(z1.size(0)).to(z1.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            return (loss_i2t + loss_t2i) / 2

    def forward(self, h, clinical_features, label=None, return_predictions=True):
        """
        Contrastive pretraining with optional classification outputs.
        """
        _, _, _, _, results_clam = self.clam(h, return_features=True)
        h_path = results_clam['features']
        
        if self.clinical_features_dim == 23:
            h_clinical = clinical_features
        else:
            _, h_clinical = self.clinical_model(clinical_features)

        h_clinical = h_clinical.unsqueeze(0)
        if self.use_laaf and label is not None:
            label = label.float().view(-1, 1)
            
            h_clinical = torch.cat([h_clinical, label], dim=1)

        z_img = F.normalize(self.proj_img(h_path), dim=1)
        z_tab = F.normalize(self.proj_tab(h_clinical), dim=1)

        # Contrastive loss
        loss = self.contrastive_loss(z_img, z_tab)

        if return_predictions:
            # Use fusion and classifier head
            
            h_clinical = h_clinical.squeeze(0)
            h_fused = self.fusion(h_path, h_clinical[:self.clinical_features_dim])  # strip LaaF label
            logits = self.classifier(h_fused)
            Y_prob = F.softmax(logits, dim=1)
            Y_hat = torch.argmax(Y_prob, dim=1)
            return loss, z_img, z_tab, Y_prob, Y_hat

        return loss, z_img, z_tab
