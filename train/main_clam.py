from __future__ import print_function

import argparse
import os

# internal imports
from utils.file_utils import save_pkl
from utils.core_utils_simul import train
from dataset.clam_dataset.dataset_generic import Multimodal_Generic_MIL_Dataset, MM_Multi_Scale_Dataset

# pytorch imports
import torch
import numpy as np
from tqdm import tqdm

def main(args, dataset):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    folds = np.arange(0, args.k)
    for i in tqdm(folds, desc='Folds'):
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path = '{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results = train(datasets, i, args, device=device)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'f1', 'f1ce', 'nll'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_mm_simul', 'mm_hierarchical'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment name for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'tiny'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['chimeraMM', 'chimeraMMMultiScale'],)
### CLAM specific options
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--no_verbose', action='store_true', default=False,
                    help='disable verbose output')
parser.add_argument('--clinical_dim', type=int, default=256,
                    help='dimension of clinical features (default: 256), use 23 for directly using CD features')
parser.add_argument('--norm', action='store_true', default=False, 
                    help='normalize the features before fusion, only for multimodal models')
parser.add_argument('--pages', type=int, nargs='+', default=None,
                    help='list of pages to use for multi-scale model, if None, all pages are used (default: None)')
parser.add_argument('--page', type=int, default=0,
                    help='page to use for single-scale model, if None, all pages are used (default: 0)')
parser.add_argument('--return_coords', action='store_true', default=False,
                    help='return coordinates of patches for multi-scale model, only for multi-scale model')
# Fine-granular learning rates & regularization
parser.add_argument('--lr_clinical', type=float, default=None,
                    help='learning rate for clinical model')
parser.add_argument('--lr_clam', type=float, default=None,
                    help='learning rate for CLAM model')
parser.add_argument('--lr_mm', type=float, default=None,
                    help='learning rate for multimodal fusion components (default: 1e-4)')

# Add new hierarchical learning rate arguments
parser.add_argument('--lr_attention', type=float, default=None,
                    help='learning rate for attention networks within CLAM hierarchy (default: same as lr_clam)')
parser.add_argument('--lr_attention_levels', type=str, default=None,
                    help='comma-separated learning rates for each attention level, e.g., "1e-4,5e-5,1e-5" for levels 0,1,2 (default: use lr_attention for all)')
parser.add_argument('--lr_fusion_net', type=float, default=None,
                    help='learning rate for fusion network layers (default: same as lr_mm)')
parser.add_argument('--lr_classifier', type=float, default=None,
                    help='learning rate for final classifier layers (default: same as lr_mm)')
parser.add_argument('--lr_level_weights', type=float, default=None,
                    help='learning rate for level weighting parameters in hierarchical CLAM (default: higher than lr_clam)')

parser.add_argument('--reg_clinical', type=float, default=None, help='weight decay for clinical model')
parser.add_argument('--reg_clam', type=float, default=None, help='weight decay for CLAM model')
parser.add_argument('--reg_mm', type=float, default=None, help='weight decay for multimodal model')
parser.add_argument('--use_scheduler', action='store_true', default=False,
                    help='use learning rate scheduler, only for CLAM models')
parser.add_argument('--multi_split', action='store_true', default=False,
                    help='train on multiple splits')
parser.add_argument('--attention_supervision_weight', type=float, default=0.0,
                    help='weight for attention supervision loss, only for CLAM models with attention supervision') # Using the tumor labels to supervise the attention weights
parser.add_argument('--top_p', type=float, default=0.3, help='For the hierarchical approach: top percentage of patches that are selected in the next layer')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_experiment(args):

    verbose = not args.no_verbose
    seed_torch(args.seed)

    assert (args.lr_clinical is not None and args.lr_clam is not None and args.lr_mm is not None) or args.lr is not None, "Please specify learning rates for all models or a single learning rate for all models."

    assert (args.reg_clinical is not None and args.reg_clam is not None and args.reg_mm is not None) or args.reg is not None, "Please specify regularization for all models or a single regularization for all models."


    if args.lr_clam is None:
        args.lr_clam = args.lr
    if args.lr_mm is None:
        args.lr_mm = args.lr
    if args.lr_clinical is None:
        args.lr_clinical = args.lr
    
    # Set defaults for new hierarchical learning rates
    if args.lr_attention is None:
        args.lr_attention = args.lr_clam
    if args.lr_fusion_net is None:
        args.lr_fusion_net = args.lr_mm
    if args.lr_classifier is None:
        args.lr_classifier = args.lr_mm
    if args.lr_level_weights is None:
        args.lr_level_weights = args.lr_clam * 2  # Higher LR for level weights as they're meta-parameters
    
    if args.reg_clam is None:
        args.reg_clam = args.reg
    if args.reg_mm is None:
        args.reg_mm = args.reg
    if args.reg_clinical is None:
        args.reg_clinical = args.reg

    if verbose: print('\nLoad Dataset')

    if args.task == 'chimeraMM':
        args.n_classes=3
        dataset = Multimodal_Generic_MIL_Dataset(
            csv_path = '/gris/gris-f/homelv/phempel/masterthesis/MMFL/data/chimera_new.csv',
                data_dir= args.data_root_dir,
                shuffle = False, 
                seed = args.seed, 
                page = args.page,
                print_info = verbose,
                label_dict = {
                    "BRS1": 0,
                    "BRS2": 1,
                    "BRS3": 2,
                },
                patient_strat= False,
                ignore=[],
                return_tumor_labels = True#args.attention_supervision_weight > 0
                )
    elif args.task == 'chimeraMMMultiScale':
        args.n_classes=3
        dataset = MM_Multi_Scale_Dataset(
            csv_path = '/gris/gris-f/homelv/phempel/masterthesis/MMFL/data/chimera_new.csv',
            return_coords = args.return_coords,
            data_dir= args.data_root_dir,
            pages = args.pages if args.pages is not None else [0, 1, 2, 3, 4],
            shuffle = False, 
            seed = args.seed, 
            print_info = verbose,
            label_dict = {
                "BRS1": 0,
                "BRS2": 1,
                "BRS3": 2,
            },
            patient_strat= False,
            ignore=[],
            return_tumor_labels = True#args.attention_supervision_weight > 0
        )
    
    
    else:
        raise NotImplementedError
    assert args.subtyping 
        
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

    if verbose: print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir), "Split directory does not exist: {}".format(args.split_dir)
    
    return dataset
            

if __name__ == "__main__":
    if not args.multi_split:
        dataset = init_experiment(args=args)
        results = main(args, dataset)
    else:
        split_dirs = [os.path.join(args.split_dir, dir_name) for dir_name in os.listdir(args.split_dir) if os.path.isdir(os.path.join(args.split_dir, dir_name))]
        results_dir = args.results_dir + '/' + args.exp_code + '_splits'
        exp_code = args.exp_code
        for i, split_dir in enumerate(split_dirs):
            print(f"\n\n==========Split {i+1}/{len(split_dirs)}==========\n\n")
            args.split_dir = split_dir
            args.exp_code = f"{exp_code}_split_{i+1}"
            args.results_dir = results_dir
            dataset = init_experiment(args=args)
            
            results = main(args, dataset)

    print("finished!")
    print("end script")
"""
CUDA_VISIBLE_DEVICES=0 python main_clam.py \
    --drop_out 0.5 --early_stopping --max_epochs 150 --weighted_sample --bag_loss ce --inst_loss svm --model_type mm_hierarchical --log_data --subtyping \
    --data_root_dir "/local/scratch/phempel/chimera/features_1536" --embed_dim 1536 --bag_weight 0.8 --B 8 --model_size tiny --task chimeraMMMultiScale --split_dir /gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection --use_scheduler --multi_split --return_coords --pages 1 2 3 \
    --exp_code Hierarchicalvacation_winner \
    --norm --lr_clam 2.5e-5 --lr_clinical 2e-5 --lr_mm 9e-5 --reg_clam 1e-3 --reg_clinical 1e-6 --reg_mm 1e-4 \
    --lr_attention_levels "4e-5,3e-5,2e-5"
"""
"""
CUDA_VISIBLE_DEVICES=1 python main_clam.py --drop_out 0.5 --early_stopping --max_epochs 150  --weighted_sample --bag_loss ce --inst_loss svm --model_type mm_hierarchical --log_data --subtyping --data_root_dir "/local/scratch/phempel/chimera/features_1536" --embed_dim 1536 --B 8 --model_size tiny --task chimeraMMMultiScale --split_dir /gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection --norm --reg_clinical 1e-6 --reg_mm 1e-4 --use_scheduler --multi_split  --return_coords --pages 1 2 3 --no_verbose --exp_code Hierarchicalvacation_winner_new_att_supervision --lr_clam 2.5e-05 --lr_clinical 2e-05 --lr_mm 9e-05 --lr_attention_levels 4e-5,3e-5,2e-5 --reg_clam 0.001 --bag_weight 0.8 --attention_supervision_weight 1
"""

"""
hier:
CUDA_VISIBLE_DEVICES=0 python main_clam.py --drop_out 0.5 --early_stopping --max_epochs 150 --weighted_sample --bag_loss ce --inst_loss svm --model_type mm_hierarchical --log_data --subtyping --data_root_dir "/local/scratch/phempel/chimera/features_1536" --embed_dim 1536 --B 8 --model_size tiny --task chimeraMMMultiScale --split_dir /gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection --k 10 --norm --reg_clinical 1e-6 --reg_mm 1e-4 --use_scheduler --multi_split --return_coords --pages 1 2 3 --no_verbose --exp_code attn_supervision_12-14-41_0_000 --lr_clam 2.5e-05 --lr_clinical 2e-05 --lr_mm 9e-05 --lr_attention_levels 4e-5,3e-5,2e-5 --reg_clam 0.001 --bag_weight 0.8
"""


"""
Non hier:

CUDA_VISIBLE_DEVICES=0 python main_clam.py --drop_out 0.5 --early_stopping --max_epochs 150 --weighted_sample --bag_loss ce --inst_loss svm --model_type clam_mm_simul --log_data --subtyping --data_root_dir "/local/scratch/phempel/chimera/features_1536" --embed_dim 1536 --B 8 --model_size tiny --task chimeraMM --split_dir /gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection --k 10 --reg_clinical 1e-6 --reg_mm 1e-4 --use_scheduler --multi_split --page 1 --norm --no_verbose --exp_code not_hier --attention_supervision_weight 75.0 --lr_clam 2.5e-05 --lr_clinical 2e-05 --lr_mm 9e-05 --reg_clam 0.001 --bag_weight 0.8
"""