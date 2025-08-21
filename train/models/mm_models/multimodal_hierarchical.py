import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cd_models.cd_nn import CD_Solo
from models.clam_models.model_multi_scale import CLAM_Hierarchical

class MultimodalHierarchical(nn.Module):
    
    def __init__(self, instance_loss_fn, *args, num_levels = 4, scale_factors=None, window_size=225, top_p=0.3, clinical_dim=32, norm = False, n_classes=3,attention_supervision_weight=0.0, **kwargs):
        
        super(MultimodalHierarchical, self).__init__()
        
        self.kind = 'hierarchical'
        
        self.clam = CLAM_Hierarchical(instance_loss_fn=instance_loss_fn, num_levels=num_levels, scale_factors=scale_factors, window_size=window_size, top_p=top_p, attention_supervision_weight=attention_supervision_weight, *args, **kwargs, n_classes=n_classes)
        
        
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

    def forward(self, h_list, clinical_features, coords=None, label=None, instance_eval=False, return_features=False, attention_only=False, slide_id=None, plot_coords = False, tumor_labels=None):
        
        # Pass the features to the CLAM model
        logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam = self.clam(h_list, coords=coords, label=label, instance_eval=instance_eval, return_features=True, attention_only=attention_only, slide_id=slide_id, plot_coords = plot_coords, tumor_label_list=tumor_labels)
        
        h_path = results_clam['features']
        attention_supervision_losses = results_clam['attention_supervision_loss']
        
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
            
        return {"CLAM": [logits_clam, Y_prob_clam, Y_hat_clam, A_raw, results_clam], "CD": [logits_clinical, Y_prob_clinical, Y_hat_clinical, None, None], "MM": [logits, Y_prob, Y_hat, None, None], "attention_supervision_loss": sum(attention_supervision_losses)}

