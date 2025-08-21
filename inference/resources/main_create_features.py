import os
import sys
import shutil

# Add paths to Python path
current_dir = os.path.dirname(__file__)
clam_path = os.path.join(current_dir, 'CLAM')
wsi_core_path = os.path.join(current_dir, 'wsi_core')
sys.path.insert(0, current_dir)
sys.path.insert(0, clam_path)
sys.path.insert(0, wsi_core_path)

# Import CLAM functions directly
from resources.feature_extraction.create_patches_fp import seg_and_patch
from resources.feature_extraction.extract_features_fp import extract_features


def create_features(wsi_path, output_dir, mask_path, stain_template_path, model_weights_path, device):
    """
    Create features from a WSI image using CLAM pipeline.
    
    Args:
        wsi_path: Path to the WSI file
        output_dir: Directory to save all outputs
        mask_path: Optional path to tissue mask
    
    Returns:
        Dictionary with paths to created files
    """
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For feature extraction, we'll use the original WSI path directly
    wsi_dir = os.path.dirname(wsi_path)
    
    # Setup output directories
    patch_save_dir = os.path.join(output_dir, 'patches') # Directory to save patch coordinates as .h5 files
    os.makedirs(patch_save_dir, exist_ok=True)
    feat_dir = os.path.join(output_dir, 'features') # Directory to save extracted features
    os.makedirs(feat_dir, exist_ok=True)


    print("Step 1: Creating patches using CLAM...")
    
    # Parameters for patching
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    

    slide_info = seg_and_patch(
        source=wsi_dir,
        patch_save_dir=patch_save_dir,
        seg_params=seg_params,
        filter_params=filter_params,
        patch_size=224,
        step_size=224,
        patch_level=1,
    )
    
    del seg_params, filter_params  # Clean up parameters to free memory

    print("Step 2: Extracting features...")

    features = extract_features(
        model_name='uni_v1',
        slide_info=slide_info,  # Pass the slide info dictionary directly
        mask_path=mask_path,
        target_image_path=stain_template_path,
        feat_dir=feat_dir,
        data_h5_dir=output_dir,
        data_slide_dir=wsi_dir,
        model_weights_path=model_weights_path,
        device=device,
        target_patch_size=224,
        batch_size=64  # Further reduced batch size to save memory
    )
    print(f"Feature extraction completed successfully.")# Features saved to: {features_path}")
    return features
    
    
    
    

    # Return paths to created files
    # slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    
    # results = {
    #     'patches_h5': os.path.join(patch_save_dir, f'{slide_id}.h5'),
    #     'features_pt': os.path.join(feat_dir, 'pt_files', f'{slide_id}.pt')
    # }
    
    # # Check which files were actually created
    # for key, path in results.items():
    #     if os.path.exists(path):
    #         print(f"✓ Created {key}: {path}")
    #     else:
    #         print(f"✗ Missing {key}: {path}")
    
    # return results



import numpy as np
import h5py

def create_features_hiearchical(wsi_path, output_dir, mask_path, stain_template_path, model_weights_path, device):
    """
    Create features from a WSI image using CLAM pipeline.
    
    Args:
        wsi_path: Path to the WSI file
        output_dir: Directory to save all outputs
        mask_path: Optional path to tissue mask
    
    Returns:
        Dictionary with paths to created files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For feature extraction, we'll use the original WSI path directly
    wsi_dir = os.path.dirname(wsi_path)
    
    # Setup output directories
    patch_save_dir = os.path.join(output_dir, 'patches') # Directory to save patch coordinates as .h5 files
    os.makedirs(patch_save_dir, exist_ok=True)
    feat_dir = os.path.join(output_dir, 'features') # Directory to save extracted features
    os.makedirs(feat_dir, exist_ok=True)


    print("Step 1: Creating patches using CLAM for each hierarchical layer...")
    
    # Parameters for patching
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    
    feature_list = []
    coords_list = []
    
    import torch
    for patch_level in [1, 2, 3]:
        print(f"Processing patch level {patch_level}...")
        slide_info = seg_and_patch(
            source=wsi_dir,
            patch_save_dir=patch_save_dir,
            seg_params=seg_params,
            filter_params=filter_params,
            patch_size=224,
            step_size=224,
            patch_level=patch_level,
        )

        print("Step 2: Extracting coordinates...")
        with h5py.File(slide_info['file_path'], 'r') as f:
            # Extract coordinates
            coords = f['coords'][:]

        # Scale the coordinates
        coords = np.array(coords.tolist(), dtype=np.float32)
        coords[:, :2] *= (1/4)**patch_level  # Scale only the first two columns (x, y)
        
        coords = torch.tensor(coords)
        coords_list.append(coords)            

        print("Step 3: Extracting features...")

        features = extract_features(
            model_name='uni_v1',
            slide_info=slide_info,  # Pass the slide info dictionary directly
            mask_path=mask_path,
            target_image_path=stain_template_path,
            feat_dir=feat_dir,
            data_h5_dir=output_dir,
            data_slide_dir=wsi_dir,
            model_weights_path=model_weights_path,
            device=device,
            target_patch_size=224,
            batch_size=64  # Further reduced batch size to save memory
        )
        feature_list.append(features)
        
    del seg_params, filter_params  # Clean up parameters to free memory
    print(f"Feature extraction completed successfully.")# Features saved to: {features_path}")
    return feature_list, coords_list


if __name__ == '__main__':
    # Test the feature creation pipeline
    output_dir = '/home/philipp/masterarbeit/chimera_baseline/test/output/interface_0/features'
    # Clear output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing output directory: {output_dir}")
    
    wsi_path = '/home/philipp/masterarbeit/chimera_baseline/test/input/interface_0/images/bladder-cancer-tissue-biopsy-wsi/2A_001_HE.tif'
    mask_path = '/home/philipp/masterarbeit/chimera_baseline/test/input/interface_0/images/tissue-mask/2A_001_HE_mask.tif'
    cd_path = '/home/philipp/masterarbeit/chimera_baseline/test/input/interface_0/chimera-clinical-data-of-bladder-cancer-patients.json'
    
    results = create_features(
        wsi_path=wsi_path,
        output_dir=output_dir,
        mask_path=mask_path,
        stain_template_path='/home/philipp/masterarbeit/chimera_baseline/chimera/resources/stain_template.png',
        model_weights_path = '/home/philipp/masterarbeit/chimera_baseline/chimera/model/pytorch_model.bin'
    )
    
    if results:
        print("\n" + "="*50)
        print("FEATURE CREATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        for key, path in results.items():
            print(f"{key}: {path}")
            
    else:
        print("Feature creation failed!")

