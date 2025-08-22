import os
import argparse

import torch
from torch.utils.data import DataLoader
import openslide

import numpy as np

from resources.feature_extraction.dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from resources.feature_extraction.models import get_encoder

def compute_w_loader(loader, model, device, verbose = 0):
	"""
	args:
		loader: data loader
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')

	all_features = []
	for _, data in enumerate(loader):
		with torch.inference_mode():	
			batch = data['img']
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu()
			all_features.append(features)
	
	# Concatenate all features
	all_features = torch.cat(all_features, dim=0)
	return all_features

def extract_features(model_name, slide_info, mask_path, target_image_path, feat_dir, data_h5_dir, data_slide_dir, model_weights_path, device, target_patch_size=224, batch_size=64):
	"""
	Simplified feature extraction for a single slide.
	
	Args:
		slide_info: Dictionary containing slide information (from seg_and_patch)
		...other parameters
	"""

	print("Loading model")
	model, img_transforms = get_encoder(model_name=model_name, target_img_size=target_patch_size, model_weights_path=model_weights_path)  
	# from resources.feature_extraction.models import get_eval_transforms
	# img_transforms = get_eval_transforms(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225],
    #                                      target_img_size = 224)
	print("Model loaded successfully.")
	_ = model.eval()
	model = model.to(device)

	loader_kwargs = {'num_workers': 0, 'pin_memory': False}  # Use 0 workers to avoid shared memory issues

	# Process the single slide
	slide_filename = slide_info['slide_id']
	
	# Handle case where slide_filename might be a pandas Series
	if hasattr(slide_filename, 'iloc'):
		slide_filename = slide_filename.iloc[0]
	
	slide_id = slide_filename.split('.tif')[0]
	bag_name = slide_id + '.h5'
	h5_file_path = os.path.join(data_h5_dir, 'patches', bag_name)
	slide_file_path = os.path.join(data_slide_dir, slide_filename)

	print(f"Processing slide: {slide_filename}")
	print(f"Slide path: {slide_file_path}")
	print(f"H5 patches path: {h5_file_path}")

	# Check if required files exist
	if not os.path.exists(slide_file_path):
		raise FileNotFoundError(f"Slide file not found: {slide_file_path}")
	if not os.path.exists(h5_file_path):
		raise FileNotFoundError(f"Patches file not found: {h5_file_path}")

	wsi = openslide.open_slide(slide_file_path)

	dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
								wsi=wsi, 
								img_transforms=img_transforms, 
								mask_path=mask_path,
								target_image_path=target_image_path,
								use_stain_norm=True)

	loader = DataLoader(dataset=dataset, batch_size=batch_size, **loader_kwargs)

	features = compute_w_loader(loader=loader, model=model, device=device, verbose=1)

	return features
	
	# # Save features
	# output_path = os.path.join(feat_dir, 'pt_files', f'{slide_id}.pt')
	# torch.save(features, output_path)
	# print(f"Features saved to: {output_path}")
	
	# return output_path



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--aug', default=False, action='store_true', help='Apply data augmentation during feature extraction')
parser.add_argument('--mask_path', type=str, default=None, help='Path to tissue mask file')
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'], help='Model name for feature extraction')
parser.add_argument('--target_image_path', type=str, default='/home/philipp/masterarbeit/chimera_baseline/chimera/resources/stain_template.png', help='Path to target image for stain normalization')
parser.add_argument('--model_weights_path', type=str, default=None, help='Path to model weights for UNI model')
args = parser.parse_args()


if __name__ == '__main__':
    import pandas as pd
    csv = pd.read_csv(args.csv_path) if args.csv_path else None
    extract_features(
		model_name=args.model_name,
		csv=csv,
		mask_path=args.mask_path,
		feat_dir=args.feat_dir,
		data_h5_dir=args.data_h5_dir,
		data_slide_dir=args.data_slide_dir,
		target_patch_size=args.target_patch_size,
		model_weights_path=args.model_weights_path,
		target_image_path=args.target_image_path,
		batch_size=args.batch_size,
	)
    print("Feature extraction completed successfully.")