import sys
import os
sys.path.append('/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cdcf')
os.chdir('/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cdcf')

from utils.fl_utils import load_data
import torch

class Args:
    def __init__(self, augmentation):
        self.seed = 1
        self.return_coords = False
        self.data_root_dir = "/local/scratch/phempel/chimera/features_1536_fixed"
        self.pages = [0, 1, 2, 3, 4]
        self.no_verbose = True
        self.split_dir = "splits/chimera_3_5_2_0.1_1"
        self.use_split_k = 0
        self.fold = 0
        self.num_stages = 2
        self.folds = 5
        self.augmentation = augmentation

# Test for del1, del2, del3
for aug in ['features_1536_fixed', 'features_1536_fixed_aug3_3_440', 'features_1536_fixed_aug4_4_450']:
    args = Args(aug)
    train_datasets, val_datasets, test_datasets = load_data(0, args)
    
    print(f"\n{aug}:")
    print(f"  Stage 0 test data_dir: {test_datasets[0].data_dir}")
    print(f"  Stage 1 test data_dir: {test_datasets[1].data_dir}")
    print(f"  Stage 0 test samples: {len(test_datasets[0])}")
    print(f"  Stage 1 test samples: {len(test_datasets[1])}")
    
    # Get first test sample from stage 0
    first_sample = test_datasets[0][0]
    features = first_sample[0]
    print(f"  Stage 0 first sample shape: {features.shape if hasattr(features, 'shape') else 'N/A'}")
    print(f"  Stage 0 first sample mean: {features.mean().item() if hasattr(features, 'mean') else 'N/A'}")
    print(f"  Stage 0 sample ID: {first_sample[-1]}")
