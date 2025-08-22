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
    def __init__(self, *args, num_levels = 4, scale_factors=None, window_size=225, top_p=0.3, **kwargs):
        """
        scale_factors: List of scale factors between consecutive levels. E.g., [4, 4] means:
                       Level 0 to 1 = x4, Level 1 to 2 = x4
        window_size: Radius (in high-res space) around selected low-res coordinates to search for matches
        top_p: Percentage of high-attention patches to select from each level (0.0 to 1.0)
        """
        # print embed_dim out of kwargs
        if 'embed_dim' in kwargs:
            embed_dim = kwargs['embed_dim']

        super().__init__(*args, **kwargs)
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

    def get_focus_features(self, h_source, coords_source, h_target, coords_target, scale_factor, level_idx):
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

        
        return h_target_focus, coords_target_focus, A_raw, top_k_coords_scaled

    def forward(self, h_list, coords=None, label=None, instance_eval=False, return_features=False, attention_only=False, slide_id=None, plot_coords=False):
        """
        h_list: list of tensors [h_0, h_1, ..., h_n] from high to low resolution,
        coords: list of coordinate tensors [coords_0, coords_1, ..., coords_n], each [N_i, 2]
        """
        assert isinstance(h_list, list), "h must be a list of tensors (one per level), but got {}".format(type(h_list))
        assert coords is not None and isinstance(coords, list), "coords must be a list per level"
        assert len(h_list) == len(coords), "h_list and coords must have same length"

        # Code is written for low-to-high resolution processing -> reverse the lists
        h_list = h_list[::-1]
        coords = coords[::-1]
        
        h_curr = h_list[0]
        coords_curr = coords[0]
        attention_maps = []

        coords_per_level = {len(h_list) : coords_curr}
        all_cords_per_level = {len(h_list): coords_curr.cpu().numpy()}  # Store coordinates for each level
        top_k_cords_per_level = {}  # Store top-k coordinates for each level
        att_sup_losses = []  # Store attention supervision losses if enabled

        for i in range(len(h_list) - 1):
            h_next = h_list[i + 1]
            coords_next = coords[i + 1]
            all_cords_per_level[len(h_list) - 1 - i] = coords_next.cpu().numpy()  # Store coordinates for next level
            
            scale = self.scale_factors[i]
            

            h_focus, coords_focus, A_raw, top_k_cords  = self.get_focus_features(h_curr, coords_curr, h_next, coords_next, scale, level_idx = i)
            attention_maps.append(A_raw)
            
            top_k_cords_per_level[len(h_list) - 1 - i] = top_k_cords.cpu().numpy()  # Store top-k coordinates for current level

            if h_focus.shape[0] == 0:
                # Fallback: no matches found, use current level
                print(f"No matches found at level {i}, using current features")
                break

            h_curr = h_focus
            coords_curr = coords_focus
            coords_per_level[len(h_list) - 1 - i] = coords_curr  # Store coordinates for current level

        
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
        
        return logits, Y_prob, Y_hat, A_raw, results_dict