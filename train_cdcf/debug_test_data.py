import sys
sys.path.append('/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cdcf')

from utils.fl_utils import load_data
import argparse

# Simulate args
class Args:
    def __init__(self):
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
        self.augmentation = "features_1536_fixed_aug3_3_440"

args = Args()

# Load data for client 0
train_datasets, val_datasets, test_datasets = load_data(0, args)

print(f"Number of stages (test datasets): {len(test_datasets)}")
print(f"\nStage 0 test dataset data_dir: {test_datasets[0].data_dir}")
print(f"Stage 1 test dataset data_dir: {test_datasets[1].data_dir}")
print(f"\nStage 0 samples: {len(test_datasets[0])}")
print(f"Stage 1 samples: {len(test_datasets[1])}")
