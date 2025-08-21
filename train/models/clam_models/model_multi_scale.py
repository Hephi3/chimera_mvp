import torch
import torch.nn.functional as F
from models.clam_models.model_clam import CLAM_SB
import numpy as np
from models.clam_models.model_clam import Attn_Net, Attn_Net_Gated
import torch.nn as nn

class CLAM_Multi_Scale(CLAM_SB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = 'multi_scale'
        
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if isinstance(h, list):
            h = torch.cat(h, dim=0)
        return super().forward(h, label=label, instance_eval=instance_eval,
                               return_features=return_features, attention_only=attention_only)

class CLAM_Hierarchical(CLAM_Multi_Scale):
    def __init__(self, *args, num_levels = 4, scale_factors=None, window_size=225, top_p=0.3, attention_supervision_weight=0.0, **kwargs):
        """
        scale_factors: List of scale factors between consecutive levels. E.g., [4, 4] means:
                       Level 0 to 1 = x4, Level 1 to 2 = x4
        window_size: Radius (in high-res space) around selected low-res coordinates to search for matches
        top_p: Percentage of high-attention patches to select from each level (0.0 to 1.0)
        """
        # print embed_dim out of kwargs
        if 'embed_dim' in kwargs:
            embed_dim = kwargs['embed_dim']

        super().__init__(attention_supervision_weight=attention_supervision_weight, *args, **kwargs)
        self.scale_factors = scale_factors
        if self.scale_factors is None:
            self.scale_factors = [4]*(num_levels - 1)
        # self.scale_factors = [1,1,1,4]
        self.window_size = window_size
        self.top_p = top_p
        self.kind = 'hierarchical'
         
        # Get size configuration
        size = self.size_dict[self.size_arg]
        
        # Create separate COMPLETE attention networks for each resolution level
        self.attention_nets = nn.ModuleList()
        
        for i in range(len(self.scale_factors) + 1):  # +1 for the final level
            # Create full attention network with feature transformation layers
            fc_layers = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(self.dropout)]
            
            if self.gate:
                attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=self.dropout, n_classes=1)
            else:
                attention_net = Attn_Net(L=size[1], D=size[2], dropout=self.dropout, n_classes=1)
                
            fc_layers.append(attention_net)
            self.attention_nets.append(nn.Sequential(*fc_layers))

    def get_focus_features(self, h_source, coords_source, h_target, coords_target, scale_factor, level_idx, tumor_labels_source=None, tumor_labels_target=None):
        """
        Select target-level features based on attention from source-level features.
        """
        # Call the attention_net properly to get attention scores
        A, h_source_transformed = self.attention_nets[level_idx](h_source)
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A.clone()
        A = F.softmax(A, dim=1)  # softmax over N

        top_k = max(1, int(self.top_p * A.shape[1]))  # Ensure at least one patch is selected
        topk_indices = torch.topk(A, top_k, dim=1)[1].squeeze(0)
        topk_coords = coords_source[topk_indices]
        
        x_center = (topk_coords[:, 0] + 224/2 - (224/4/2)) * scale_factor
        y_center = (topk_coords[:, 1] + 224/2 - (224/4/2)) * scale_factor
        centers = torch.stack([x_center, y_center], dim=1)  # [k, 2]
        
        # Compute distances in target space
        matched_indices = []
        for c in centers:
            d = torch.norm(coords_target - c.unsqueeze(0), dim=1)
            nearby = torch.where(d < self.window_size * scale_factor)
            if len(nearby[0]) == 0:
                print(f"No matches found for center {c} at level {level_idx}")
                print()
            nearby = nearby[0]
            matched_indices.append(nearby)

        if matched_indices:
            matched_indices = torch.cat(matched_indices).unique()
            h_target_focus = h_target[matched_indices]
            coords_target_focus = coords_target[matched_indices]
        else:
            print("No matches found at level", level_idx)
            h_target_focus = torch.empty(0, h_target.shape[1], device=h_target.device)
            coords_target_focus = torch.empty(0, 2, device=coords_target.device)

        top_k_coords_scaled = torch.stack([topk_coords[:, 0] * scale_factor, topk_coords[:, 1] * scale_factor], dim=1)
        
        if tumor_labels_target is not None and len(tumor_labels_target) > 0:
            
            topk_tumor_labels = tumor_labels_source[topk_indices] if tumor_labels_source is not None else None
            topk_A = A.squeeze(0)[topk_indices]  # Get attention scores for selected patches
        
            # Ensure tumor_labels is a tensor and matches coords_target
            if not isinstance(tumor_labels_target, torch.Tensor):
                tumor_labels_target = torch.tensor(tumor_labels_target, device=h_target.device)
            assert len(tumor_labels_target) == len(coords_target), \
                f"Length mismatch: tumor_labels ({len(tumor_labels_target)}) vs coords_target ({len(coords_target)})"
            if matched_indices.numel() > 0:
                tumor_labels_focus = tumor_labels_target[matched_indices]
            else:
                tumor_labels_focus = torch.empty(0, dtype=tumor_labels_target.dtype, device=tumor_labels_target.device)
            
            return h_target_focus, coords_target_focus, A_raw, top_k_coords_scaled, tumor_labels_focus, topk_A, topk_tumor_labels
        
        return h_target_focus, coords_target_focus, A_raw, top_k_coords_scaled, None, None, None

    def forward(self, h_list, coords=None, label=None, instance_eval=False, return_features=False, attention_only=False, slide_id=None, plot_coords=False, tumor_label_list=None):
        """
        h_list: list of tensors [h_0, h_1, ..., h_n] from high to low resolution,
        coords: list of coordinate tensors [coords_0, coords_1, ..., coords_n], each [N_i, 2]
        """
        assert isinstance(h_list, list), "h must be a list of tensors (one per level), but got {}".format(type(h_list))
        assert coords is not None and isinstance(coords, list), "coords must be a list per level"
        assert len(h_list) == len(coords), "h_list and coords must have same length"
        if tumor_label_list is not None:
            assert len(coords) == len(tumor_label_list), f"tumor_label_list must have the same length as h_list for slide {slide_id}"
            for i in range(len(coords)):
                assert len(coords[i]) == len(tumor_label_list[i]) or slide_id in ['2A_017_HE', '2A_031_HE', '2A_060_HE', '2A_074_HE', '2A_098_HE', '2A_127_HE', '2A_141_HE', '2A_143_HE', '2A_145_HE', '2A_157_HE', '2B_288_HE', '2B_365_HE', '2A_001_HE'], f"tumor_label_list[{i}] must have the same length as coords[{i}] but got {len(tumor_label_list[i])} vs {len(coords[i])} and slide_id {slide_id}"

        # Code is written for low-to-high resolution processing -> reverse the lists
        h_list = h_list[::-1]
        coords = coords[::-1]
        tumor_label_list = tumor_label_list[::-1] if tumor_label_list is not None else None
        
        h_curr = h_list[0]
        coords_curr = coords[0]
        tumor_labels_curr = tumor_label_list[0] if tumor_label_list is not None else None
        attention_maps = []

        coords_per_level = {len(h_list) : coords_curr}
        all_cords_per_level = {len(h_list): coords_curr.cpu().numpy()}  # Store coordinates for each level
        top_k_cords_per_level = {}  # Store top-k coordinates for each level
        att_sup_losses = []  # Store attention supervision losses if enabled

        for i in range(len(h_list) - 1):
            h_next = h_list[i + 1]
            coords_next = coords[i + 1]
            all_cords_per_level[len(h_list) - 1 - i] = coords_next.cpu().numpy()  # Store coordinates for next level
            
            if tumor_label_list is not None:
                tumor_labels_next = tumor_label_list[i + 1] if tumor_label_list is not None else None
                assert len(tumor_labels_next) == len(coords_next) or len(tumor_labels_next) == 0 or slide_id == "2A_001_HE", \
                    f"tumor_labels_next must have the same length as coords_next, but got {len(tumor_labels_next)} vs {len(coords_next)}"
                if slide_id == "2A_001_HE":
                    tumor_labels_next = None  # Special case for this slide, no tumor labels available
                    tumor_labels_curr = None
            else:
                tumor_labels_next = None
                tumor_labels_curr = None
            scale = self.scale_factors[i]
            

            h_focus, coords_focus, A_raw, top_k_cords, tumor_labels_focus, top_A, top_k_tumor_labels  = self.get_focus_features(h_curr, coords_curr, h_next, coords_next, scale, level_idx = i, tumor_labels_source=tumor_labels_curr, tumor_labels_target=tumor_labels_next)
            attention_maps.append(A_raw)
            
            # Attention supervision: if enabled, compute loss based on attention maps
            if self.attention_supervision_weight > 0.0 and top_k_tumor_labels is not None:
                # Ensure tumor_labels_list is a list of tensors
                assert len(top_k_tumor_labels) == len(top_A), \
                    f"tumor_labels_list must be a list with the same length as A_raw, but got {len(top_k_tumor_labels)} vs {len(top_A)}, A_raw shape: {top_A.shape}"
                att_sup_loss = self.attention_supervision_loss(top_A, top_k_tumor_labels, slide_id)
                att_sup_losses.append(att_sup_loss)
            
            top_k_cords_per_level[len(h_list) - 1 - i] = top_k_cords.cpu().numpy()  # Store top-k coordinates for current level

            if h_focus.shape[0] == 0:
                # Fallback: no matches found, use current level
                print(f"No matches found at level {i}, using current features")
                break

            h_curr = h_focus
            coords_curr = coords_focus
            coords_per_level[len(h_list) - 1 - i] = coords_curr  # Store coordinates for current level
            tumor_labels_curr = tumor_labels_focus

        
        # Apply final attention using parent class functionality
        A, h = self.attention_nets[-1](h_curr)
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A.clone()
        attention_maps.append(A_raw)
        
        if attention_only:
            return A_raw
            
        A = F.softmax(A, dim=1)  # softmax over N
        
        # Instance-level evaluation
        results_dict = {}
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                # print(f"Inst_labels: {inst_labels}")
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
            
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
                           'inst_preds': np.array(all_preds)}
            
        # Final attention-weighted features
        M = torch.mm(A, h)
        logits = self.classifiers(M)  # Use self.classifiers from parent class
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if return_features:
            results_dict.update({'features': M, 'attention_maps': attention_maps})
        
        if self.attention_supervision_weight > 0.0 and att_sup_losses:
            # Average attention supervision loss across levels
            results_dict['attention_supervision_loss'] = att_sup_losses
        else:
            results_dict['attention_supervision_loss'] = [0]
        
        return logits, Y_prob, Y_hat, A_raw, results_dict