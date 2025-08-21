import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

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
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, 
        attention_supervision_weight=0.0):
        
        super().__init__()
        self.kind = 'img'
        self.attention_supervision_weight = attention_supervision_weight
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
    
    def attention_supervision_loss(self, attention_weights, tumor_labels, slide_id=None):
        """
        Supervise attention to focus on tumor regions.
        attention_weights: [N] or [1, N] - raw attention scores from attention network
        tumor_labels: [N] - binary mask for tumor regions (1=tumor, 0=normal)
        """
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.squeeze()
        
        # Ensure tensors are on the same device
        tumor_labels = tumor_labels.to(attention_weights.device)
        
        # Convert raw attention scores to probabilities
        attention_probs = F.softmax(attention_weights, dim=0)
        
        # Create target distribution from tumor mask
        tumor_target = tumor_labels.float()
        
        # print(1/0)
        # Avoid division by zero
        if tumor_target.sum() == 0:
            # If no tumor patches, return zero loss
            return torch.tensor(0.0, device=attention_weights.device, requires_grad=True)
        
        # Normalize tumor mask to create probability distribution
        tumor_target = tumor_target / tumor_target.sum()
        
        # KL divergence loss: KL(attention_probs || tumor_target)
        # If attention_probs has no shape but is a single value, it should be ensqeezed
        if attention_probs.dim() == 0:
            attention_probs = attention_probs.unsqueeze(0)
        # print("SHAPES:", attention_probs.shape, tumor_target.shape, attention_probs)
        if attention_probs.shape[0] != tumor_target.shape[0]:
            print("!!!! WARNING: Attention probabilities shape does not match tumor target shape!!!! att, tum:", attention_probs.shape, tumor_target.shape, "slide_id:", slide_id)
            # Optionally print a warning or debug info here
            return torch.tensor(0.0, device=attention_weights.device, requires_grad=True)
        assert attention_probs.shape == tumor_target.shape, f"Attention probabilities shape {attention_probs.shape} does not match tumor target shape {tumor_target.shape}"
        kl_loss = F.kl_div(
            attention_probs.log(), 
            tumor_target + 1e-8,  # Add small epsilon for numerical stability
            reduction='batchmean'
        )
        # print("CALCULATED KL LOSS:", kl_loss.item())
        return kl_loss
    
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

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, tumor_labels=None):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
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
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        # Add attention supervision loss if tumor mask is provided and we're training
        if tumor_labels is not None and len(tumor_labels) > 0 and self.training and self.attention_supervision_weight > 0:
            att_sup_loss = self.attention_supervision_loss(A_raw.squeeze(), tumor_labels)
            results_dict['attention_supervision_loss'] = att_sup_loss * self.attention_supervision_weight
        else:
            results_dict['attention_supervision_loss'] = 0
        return logits, Y_prob, Y_hat, A_raw, results_dict