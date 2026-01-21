path1 = "/gris/gris-f/homelv/phempel/masterthesis/test/tmp/features/features/pt_files/2A_002.pt"
path2 = "/local/scratch/phempel/chimera/features_1536/features_CLAM_page1/2A_002_HE.pt"
# path2 = "/gris/gris-f/homelv/phempel/masterthesis/test/tmp/features/features/pt_files/2A_002 copy.pt"

path3 = "/gris/gris-f/homelv/phempel/masterthesis/test/tmp/features/patches/2A_002.h5"
path4 = "/local/scratch/phempel/chimera/test/features_CLAM_page1/h5_files/2A_002_HE.h5"

import os
import h5py
import torch
def compare_features(path1, path2):
    """
    Compare two feature files and print differences.
    """
    if not os.path.exists(path1):
        print(f"File not found: {path1}")
        return
    if not os.path.exists(path2):
        print(f"File not found: {path2}")
        return
    if not os.path.exists(path3):
        print(f"File not found: {path3}")
        return
    if not os.path.exists(path4):
        print(f"File not found: {path4}")
        return
    
    patches1 = h5py.File(path3, 'r')
    patches2 = h5py.File(path4, 'r')
    
    print(f"Shapes: {os.path.basename(path3)}: {patches1['coords'].shape}, {os.path.basename(path4)}: {patches2['coords'].shape}")
    
    # First 10 elements
    print("First 10 elements:")
    print(f"{os.path.basename(path3)}: {patches1['coords'][:10]}")
    print(f"{os.path.basename(path4)}: {patches2['coords'][:10]}")
    
    
    features1 = torch.load(path1)
    features2 = torch.load(path2)
    
    print(features1)
    
    print(f"Shapes: {os.path.basename(path1)}: {features1.shape}, {os.path.basename(path2)}: {features2.shape}")
    
    # First 10 elements
    print("First 10 elements:")
    print(f"{os.path.basename(path1)}: {features1[:10]}")
    print(f"{os.path.basename(path2)}: {features2[:10]}")
    

    
if __name__ == "__main__":
    compare_features(path1, path2)