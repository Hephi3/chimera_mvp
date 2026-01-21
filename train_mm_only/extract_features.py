import torch
import numpy as np
import h5py
import os

from dataset.dataset_iterator import root_iter, id_to_filename
from data.feature_extraction.create_patches_fp import seg_and_patch
from data.feature_extraction.extract_features_fp import extract_features

def create_features_hiearchical(wsi_path, output_dir, mask_path, stain_template_path, model_weights_path, device, id):
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
    for patch_level in [1, 2, 3]:
        print(f"Processing patch level {patch_level}...")
        slide_info = seg_and_patch(
            source=wsi_path,
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

        if not os.path.exists(os.path.join(output_dir, f'coordinates_page{patch_level}_scaled')):
            os.makedirs(os.path.join(output_dir, f'coordinates_page{patch_level}_scaled'))
        np.save(os.path.join(output_dir, f'coordinates_page{patch_level}_scaled', f"{id_to_filename(id)}.npy"), coords)   

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
        pt_filename = f"{id_to_filename(id)}.pt"
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(os.path.join(output_dir, f'features_CLAM_page{patch_level}')):
            os.makedirs(os.path.join(output_dir, f'features_CLAM_page{patch_level}'))
        torch.save(features, os.path.join(output_dir, f'features_CLAM_page{patch_level}', pt_filename))
        
    del seg_params, filter_params  # Clean up parameters to free memory
    print(f"Feature extraction completed successfully.")# Features saved to: {features_path}")
    return feature_list, coords_list

if __name__ == "__main__":
    device = "cuda:5" if torch.cuda.is_available() else "cpu"

    root_iterator = root_iter()
    for id, patient_data in root_iterator:
        wsi_path = patient_data["hist"]
        mask_path = patient_data["hist_mask"]

        img_features_list, coords = create_features_hiearchical(
            wsi_path=wsi_path,
            output_dir= "data/features", 
            mask_path=mask_path,
            stain_template_path="data/stain_template.png",
            model_weights_path="data/pytorch_model.bin",
            device=device,
            id=id
    )