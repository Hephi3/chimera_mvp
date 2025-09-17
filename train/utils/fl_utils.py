import torch
import torch.nn as nn
from flwr.common import Context
from typing import List
from collections import OrderedDict
import numpy as np

from dataset.clam_dataset.dataset_generic import MM_Multi_Scale_Dataset
from models.mm_models.multimodal_hierarchical import MultimodalHierarchical


def load_data(partition_id, args):
    # Ensure deterministic dataset loading
    import random
    import numpy as np
    import torch
    
    # Re-seed before dataset operations to ensure consistency
    seed = args.seed + partition_id  # Add partition_id to avoid identical seeds across clients
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = MM_Multi_Scale_Dataset(
        csv_path = '/gris/gris-f/homelv/phempel/masterthesis/MMFL/data/chimera_new.csv',
        return_coords = args.return_coords,
        data_dir= args.data_root_dir,
        pages = args.pages if args.pages is not None else [0, 1, 2, 3, 4],
        shuffle = False,
        seed = args.seed,  # Use original seed for dataset consistency
        print_info = not args.no_verbose,
        label_dict = {
            "BRS1": 0,
            "BRS2": 1,
            "BRS3": 2,
        },
        patient_strat= False,
        ignore=[]
    )

    train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path = '{}/splits_{}.csv'.format(args.split_dir, partition_id + args.use_split_k))
    return train_dataset, val_dataset, test_dataset

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_model(args, device):
    # Ensure deterministic model initialization - identical to MVP approach
    import random
    import numpy as np
    import torch
    import os
    
    # Use identical seeding strategy as MVP
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Create model
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None:
        model_dict.update({"size_arg": args.model_size})
    
    if args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    
    # if args.norm:
    #     model_dict.update({'norm': True})
    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    model = MultimodalHierarchical(**model_dict, instance_loss_fn=instance_loss_fn, num_levels=len(args.pages), clinical_dim=args.clinical_dim,
            norm=args.norm, 
            top_p=args.top_p)
    model = model.to(device)
    return model