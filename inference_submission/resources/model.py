import torch
import torch.nn as nn
import torch.nn.functional as F
from resources.model_img import CLAM_SB, CLAM_Hierarchical
from resources.model_cd import CD_Solo

class MultimodalFusionSimultaneousModel(nn.Module):
    def __init__(self, gate=True, size_arg="tiny", dropout=0.25, 
             k_sample=8, n_classes=3, instance_loss_fn=None, 
             subtyping=False, clinical_dim=32, embed_dim=1024, 
             norm=False, skip_connection=False,
             **kwargs):
        self.skip_connection = skip_connection
        super(MultimodalFusionSimultaneousModel, self).__init__()
        
        # Initialize CLAM_SB model with unpacked arguments
        self.clam = CLAM_SB(gate=gate, size_arg=size_arg, dropout=dropout,
                       k_sample=k_sample, n_classes=n_classes,
                       instance_loss_fn=instance_loss_fn,
                       subtyping=subtyping, embed_dim=embed_dim)

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
        self.classifier = nn.Linear(128 + (self.clam_features_dim if self.skip_connection else 0), n_classes)
        
    def fusion(self, h_path, h_clinical):
        """
        Concatenate image and clinical features.
        """
        h_clinical = h_clinical.unsqueeze(0)
        h_combined = torch.cat([h_path, h_clinical], dim=1)
        h_fused = self.fusion_net(h_combined)
        if self.skip_connection:
            h_fused = torch.cat([h_fused, h_path], dim=1)
        return h_fused
    
    def predict(self, h, clinical_features, return_features=False):
        results = self.forward(h, clinical_features,return_features)
        return results["MM"][1][-1]
        
    
    def forward(self, h, clinical_features):
        """
        h: Image features from dataloader
        clinical_features: Clinical data features
        """
        
        # Process WSI data through CLAM (gets features instead of logits due to Identity)
        logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam = self.clam(h, return_features=True)
        
        # Get image features from CLAM results
        h_path = results_clam['features']  # This is the M matrix (aggregated features)
        
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
            logits_clinical = logits_clinical.unsqueeze(0)  # Ensure clinical features
            
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
            
        return {"CLAM": [logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam], "CD": [logits_clinical, Y_prob_clinical, Y_hat_clinical, None, None], "MM": [logits, Y_prob, Y_hat, None, None]}




class MultimodalHierarchical(nn.Module):
    
    def __init__(self, instance_loss_fn, *args, num_levels = 4, scale_factors=None, window_size=225, top_p=0.3, clinical_dim=32, norm = False, n_classes=3, use_auxiliary_loss=False, **kwargs):
        
        super(MultimodalHierarchical, self).__init__()
        
        self.clam = CLAM_Hierarchical(instance_loss_fn=instance_loss_fn, num_levels=num_levels, scale_factors=scale_factors, window_size=window_size, top_p=top_p, use_auxiliary_loss=use_auxiliary_loss, *args, **kwargs, n_classes=n_classes)        
        
        size_arg = kwargs.get('size', 'tiny')
        dropout = kwargs.get('dropout', 0.5)  # Default dropout value
        self.clam_features_dim = self.clam.size_dict[size_arg][1]
        
        self.clinical_model = CD_Solo(input_dim=23, dropout=dropout, return_features=True) 
        self.clinical_features_dim = clinical_dim
        
        self.norm = norm
        if self.norm:
            self.clam_norm = nn.LayerNorm(self.clam_features_dim)
            self.clinical_norm = nn.LayerNorm(self.clinical_features_dim)
            self.clam_weight = nn.Parameter(torch.ones(1))
            self.clinical_weight = nn.Parameter(torch.ones(1))
        
        self.loss_weight_mm = nn.Parameter(torch.tensor(0.0))  # log(1)
        self.loss_weight_clam = nn.Parameter(torch.tensor(0.0))
        self.loss_weight_cd = nn.Parameter(torch.tensor(0.0))
        
        self.fusion_dim = self.clam_features_dim + self.clinical_features_dim
        
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

    def forward(self, h_list, clinical_features, coords):
        
        # Pass the features to the CLAM model
        logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam = self.clam(h_list, coords=coords, return_features=True)
        
        h_path = results_clam['features']
        
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
            
        return {"CLAM": [logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam], "CD": [logits_clinical, Y_prob_clinical, Y_hat_clinical, None, None], "MM": [logits, Y_prob, Y_hat, None, None]}
        
        return clam_output

    def predict(self, h_list, clinical_features, coords):
        results = self.forward(h_list, clinical_features, coords)
        return results["MM"][1][-1]