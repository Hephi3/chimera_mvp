import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.kind = 'img'
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384], "tiny": [embed_dim, 256, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.dropout = dropout
        self.size_arg = size_arg
        self.gate = gate
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = min(self.k_sample, A.shape[0])  # ensure k_sample does not exceed the number of instances
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A,k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = min(self.k_sample, A.shape[0])  # ensure k_sample does not exceed the number of instances
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, return_features=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict





import os
import tifffile
from PIL import Image

def open_image_scaled_to_level(id, page, level: int = 0) -> Image.Image:
    """
    Open an image for a given patient ID and page, scaled to the specified level.
    The level corresponds to the downsampling factor of 2^level.
    """
    root_dir = "/local/scratch/chimera/task2new/data"
    
    def id_to_filename(id, he = False):
        filename = f"2{'A' if int(id)<200 else 'B'}_{id}"
        if he:
            filename += "_HE"
        return filename
    patient_dir = id_to_filename(id)    
    patient_path = os.path.join(root_dir, patient_dir)
    image_path = os.path.join(patient_path, f"{patient_dir}_HE.tif")
    
    assert os.path.exists(image_path), f"Image file {image_path} does not exist"
    
    with tifffile.TiffFile(image_path) as tif:
        if page >= len(tif.pages):
            print(f"Page {page} is not valid. Page should be < {len(tif.pages)}")
            return None
        arr = tif.pages[page].asarray()
        img = Image.fromarray(arr)
    
    # Get width and height of the image of level level
    with tifffile.TiffFile(image_path) as tif:
        if level >= len(tif.pages):
            print(f"Level {level} is not valid. Level should be < {len(tif.pages)}")
            return None
        target_page = tif.pages[level]
        target_width = target_page.imagewidth
        target_height = target_page.imagelength

    img = img.resize((target_width, target_height), Image.LANCZOS)
    
    return img



















class CLAM_Multi_Scale(CLAM_SB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = 'multi_scale'
        
    def forward(self, h, return_features=False):
        if isinstance(h, list):
            h = torch.cat(h, dim=0)
        return super().forward(h, return_features=return_features)

class CLAM_Hierarchical(CLAM_Multi_Scale):
    def __init__(self, *args, num_levels = 4, scale_factors=None, window_size=225, top_p=0.3, use_auxiliary_loss=False, **kwargs):
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
        # Optional: Add auxiliary classifiers for intermediate supervision
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_classifiers = nn.ModuleList() if self.use_auxiliary_loss else None
        
        for i in range(len(self.scale_factors) + 1):  # +1 for the final level
            # Create full attention network with feature transformation layers
            fc_layers = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(self.dropout)]
            
            if self.gate:
                attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=self.dropout, n_classes=1)
            else:
                attention_net = Attn_Net(L=size[1], D=size[2], dropout=self.dropout, n_classes=1)
                
            fc_layers.append(attention_net)
            self.attention_nets.append(nn.Sequential(*fc_layers))
            
            # Add auxiliary classifier for this level (except the last one which uses main classifier)
            if self.use_auxiliary_loss and i < len(self.scale_factors):
                aux_classifier = nn.Linear(size[1], self.n_classes)
                self.auxiliary_classifiers.append(aux_classifier)

    def get_focus_features(self, h_source, coords_source, h_target, coords_target, scale_factor, level_idx):
        """
        Select target-level features based on attention from source-level features.
        Returns intermediate predictions if auxiliary loss is enabled.
        """
        # Call the attention_net properly to get attention scores
        A, h_source_transformed = self.attention_nets[level_idx](h_source)
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A.clone()
        A = F.softmax(A, dim=1)  # softmax over N
        
        # Compute auxiliary prediction if enabled (for intermediate supervision)
        aux_logits = None
        if self.use_auxiliary_loss and level_idx < len(self.auxiliary_classifiers):
            # Use attention-weighted features for auxiliary prediction
            M_aux = torch.mm(A, h_source_transformed)
            aux_logits = self.auxiliary_classifiers[level_idx](M_aux)

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
            nearby = nearby[0]
            matched_indices.append(nearby)

        if matched_indices:
            matched_indices = torch.cat(matched_indices).unique()
            h_target_focus = h_target[matched_indices]
            coords_target_focus = coords_target[matched_indices]
        else:
            h_target_focus = torch.empty(0, h_target.shape[1], device=h_target.device)
            coords_target_focus = torch.empty(0, 2, device=coords_target.device)

        top_k_coords_scaled = torch.stack([topk_coords[:, 0] * scale_factor, topk_coords[:, 1] * scale_factor], dim=1)
        
        return h_target_focus, coords_target_focus, A_raw, top_k_coords_scaled, aux_logits

    def forward(self, h_list, coords=None, return_features=False):
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
        auxiliary_predictions = []  # Store auxiliary predictions if enabled

        for i in range(len(h_list) - 1):
            h_next = h_list[i + 1]
            coords_next = coords[i + 1]
            all_cords_per_level[len(h_list) - 1 - i] = coords_next.cpu().numpy()  # Store coordinates for next level
            
            scale = self.scale_factors[i]

            h_focus, coords_focus, A_raw, top_k_cords, aux_logits = self.get_focus_features(h_curr, coords_curr, h_next, coords_next, scale, level_idx = i)
            attention_maps.append(A_raw)
            
            # Store auxiliary prediction if available
            if aux_logits is not None:
                auxiliary_predictions.append(aux_logits)
            
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
    
            
        A = F.softmax(A, dim=1)  # softmax over N
        
        # Instance-level evaluation
        results_dict = {}
            
        # Final attention-weighted features
        M = torch.mm(A, h)
        logits = self.classifiers(M)  # Use self.classifiers from parent class
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if return_features:
            results_dict.update({'features': M, 'attention_maps': attention_maps})
            # Include auxiliary predictions if available
            if auxiliary_predictions:
                results_dict['auxiliary_predictions'] = auxiliary_predictions
            
        return logits, Y_prob, Y_hat, A_raw, results_dict