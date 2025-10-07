import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch.optim as optim


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
            indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    # Detect batch type based on the first item's structure
    first_item = batch[0]
    
    # Case 1: Multi-Scale dataset with features organized by page
    if isinstance(first_item[0], list):
        # Multi-scale features are now organized as a list of tensors
        num_scales = len(first_item[0])
        
        collated = []
        
        # Collect features for each scale
        for scale_idx in range(num_scales):
            collated.append(torch.cat([item[0][scale_idx] for item in batch], dim=0))
            
        labels = torch.LongTensor([item[1] for item in batch])
        
        coords = []
        for scale_idx in range(num_scales):
            coords.append(torch.cat([item[2][scale_idx] for item in batch], dim=0))
        
        if len(first_item) >= 5: # Id
            clinical_data = [item[3] for item in batch]
            
            if all(isinstance(cd, torch.Tensor) for cd in clinical_data):
                clinical = torch.stack(clinical_data)
            else:
                clinical = clinical_data  # Keep as list if mixed types
            return [collated, labels, coords, clinical, first_item[4]]  # Assuming first_item[4] is the slide ID
    else:
        raise NotImplementedError("Unsupported batch format")


def get_split_loader(split_dataset, training=False, weighted=False, device=None, args=None):
    """
            return either the validation loader or training loader 
    """

    # For single client federated learning, use identical behavior as MVP
    if args and args.num_clients == 1:
        # Use MVP-identical behavior with explicit generator for reproducible sampling
        kwargs = {'num_workers': 8, 'pin_memory': False,
                  'batch_size': 1} if device.type == "cuda" else {}
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                # Use explicit generator to ensure identical sampling order as MVP
                generator = torch.Generator()
                generator.manual_seed(args.seed)
                loader = DataLoader(split_dataset, sampler=WeightedRandomSampler(
                    weights, len(weights), generator=generator), collate_fn=collate_MIL, **kwargs)
            else:
                # Use explicit generator to ensure identical sampling order as MVP
                generator = torch.Generator()
                generator.manual_seed(args.seed)
                loader = DataLoader(split_dataset, sampler=RandomSampler(
                    split_dataset, generator=generator), collate_fn=collate_MIL, **kwargs)
        else:
            loader = DataLoader(split_dataset, sampler=SequentialSampler(
                split_dataset), collate_fn=collate_MIL, **kwargs)
        return loader
    
    # For multi-client federated learning, use enhanced deterministic behavior
    def worker_init_fn(worker_id):
        import torch
        import numpy as np
        import random
        # Each worker gets a different but deterministic seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    kwargs = {
        'num_workers': 8,
        'pin_memory': False,
        'worker_init_fn': worker_init_fn,
        'batch_size': 1} if device.type == "cuda" else {}
    
    # Use args.seed if provided, otherwise fall back to default
    seed = args.seed if args is not None else 42
    
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            # Use explicit generator for reproducible sampling - seed 1 to match federated version
            generator = torch.Generator()
            generator.manual_seed(1)
            loader = DataLoader(split_dataset, sampler=WeightedRandomSampler(
                weights, len(weights), generator=generator), collate_fn=collate_MIL, **kwargs)
        else:
            # Use explicit generator for reproducible sampling - seed 1 to match federated version
            generator = torch.Generator()
            generator.manual_seed(1)
            loader = DataLoader(split_dataset, sampler=RandomSampler(
                split_dataset, generator=generator), collate_fn=collate_MIL, **kwargs)
    else:
        loader = DataLoader(split_dataset, sampler=SequentialSampler(
            split_dataset), collate_fn=collate_MIL, **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
            all_params = set(model.parameters())
            clam_params = set(model.clam.parameters())
            clinical_params = set(model.clinical_model.parameters())
            if hasattr(model, 'fusion_net'):
                mm_params = set(list(model.fusion_net.parameters()) + list(model.classifier.parameters()))
            else:
                mm_params = set()
            
            # Special handling for hierarchical models with per-level learning rates
            if hasattr(args, 'lr_attention_levels') and args.lr_attention_levels is not None:
                # Parse attention level learning rates
                attention_level_lrs = [float(lr.strip()) for lr in args.lr_attention_levels.split(',')]
                
                # Group attention network parameters by level
                param_groups = []
                
                # Add clinical parameters
                param_groups.append({
                    'params': list(clinical_params), 
                    'lr': args.lr_clinical, 
                    'weight_decay': args.reg_clinical,
                    'name': 'clinical'
                })
                
                # Add attention network parameters per level
                attention_params_handled = set()
                if hasattr(model.clam, 'attention_nets'):
                    for level_idx, attention_net in enumerate(model.clam.attention_nets):
                        level_params = set(attention_net.parameters())
                        lr = attention_level_lrs[level_idx] if level_idx < len(attention_level_lrs) else args.lr_attention
                        param_groups.append({
                            'params': list(level_params),
                            'lr': lr,
                            'weight_decay': args.reg_clam,
                            'name': f'attention_level_{level_idx}'
                        })
                        attention_params_handled.update(level_params)


                # Remaining CLAM parameters (excluding attention networks)
                remaining_clam_params = clam_params - attention_params_handled
                if remaining_clam_params:
                    param_groups.append({
                        'params': list(remaining_clam_params),
                        'lr': args.lr_clam,
                        'weight_decay': args.reg_clam,
                        'name': 'clam_other'
                    })
                
                # Fusion network parameters
                if hasattr(model, 'fusion_net') and hasattr(args, 'lr_fusion_net') and args.lr_fusion_net is not None:
                    fusion_params = set(model.fusion_net.parameters())
                    param_groups.append({
                        'params': list(fusion_params),
                        'lr': args.lr_fusion_net,
                        'weight_decay': args.reg_mm,
                        'name': 'fusion_net'
                    })
                    mm_params = mm_params - fusion_params
                
                # Classifier parameters
                if hasattr(model, 'classifier') and hasattr(args, 'lr_classifier') and args.lr_classifier is not None:
                    classifier_params = set(model.classifier.parameters())
                    param_groups.append({
                        'params': list(classifier_params),
                        'lr': args.lr_classifier,
                        'weight_decay': args.reg_mm,
                        'name': 'classifier'
                    })
                    mm_params = mm_params - classifier_params
                
                # Level weights (if they exist)
                if hasattr(model.clam, 'level_weights') and hasattr(args, 'lr_level_weights') and args.lr_level_weights is not None:
                    level_weight_params = {model.clam.level_weights}
                    param_groups.append({
                        'params': list(level_weight_params),
                        'lr': args.lr_level_weights,
                        'weight_decay': args.reg_clam * 0.1,  # Lower weight decay for meta-parameters
                        'name': 'level_weights'
                    })
                    clam_params = clam_params - level_weight_params
                
                # Remaining MM parameters
                if mm_params:
                    param_groups.append({
                        'params': list(mm_params),
                        'lr': args.lr_mm,
                        'weight_decay': args.reg_mm,
                        'name': 'mm_other'
                    })
                
                # Handle any remaining parameters
                explicitly_handled = clinical_params | attention_params_handled | remaining_clam_params
                if hasattr(model, 'fusion_net'):
                    explicitly_handled |= set(model.fusion_net.parameters())
                if hasattr(model, 'classifier'):
                    explicitly_handled |= set(model.classifier.parameters())
                if hasattr(model.clam, 'level_weights'):
                    explicitly_handled.add(model.clam.level_weights)
                
                extra_params = all_params - explicitly_handled
                if extra_params:
                    param_groups.append({
                        'params': list(extra_params),
                        'lr': args.lr,
                        'weight_decay': args.reg,
                        'name': 'extra'
                    })
                
                optimizer = torch.optim.Adam(param_groups)
                
                # Print parameter group info for debugging
                if hasattr(args, 'no_verbose') and not args.no_verbose:
                    print("\nOptimizer parameter groups:")
                    for i, group in enumerate(param_groups):
                        print(f"  Group {i} ({group.get('name', 'unnamed')}): {len(group['params'])} params, lr={group['lr']:.2e}, wd={group['weight_decay']:.2e}")
                
            else:
                # Default behavior for other multimodal models
                explicitly_handled = clam_params | clinical_params | mm_params
                extra_params = all_params - explicitly_handled
                
                optimizer = torch.optim.Adam([
                    {'params': list(clam_params), 'lr': args.lr_clam, 'weight_decay': args.reg_clam},
                    {'params': list(clinical_params), 'lr': args.lr_clinical, 'weight_decay': args.reg_clinical},
                    {'params': list(mm_params), 'lr': args.lr_mm, 'weight_decay': args.reg_mm},
                    {'params': list(extra_params), 'lr': args.lr, 'weight_decay': args.reg}
                ])
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N/len(dataset.slide_cls_ids[c])
                        for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)