# internal imports
from .wsi_core.WholeSlideImage import WholeSlideImage
# other imports
import os
import numpy as np
import time
import argparse
import pandas as pd

def segment(WSI_object, seg_params = None, filter_params = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):

	file_path = WSI_object.process_contours(**kwargs)
	return file_path


def seg_and_patch(source, patch_save_dir, patch_size=256, step_size=256, 
				  seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8}, 
				  patch_level=0):
	"""
	Simplified function to process a single slide (WSI image).
	
	Args:
		source: Path to the slide file or directory containing one slide
		patch_save_dir: Directory to save patches
		...other parameters for segmentation and patching
	"""
	
	# Handle both file path and directory input
	if os.path.isfile(source):
		slide_path = source
		slide_name = os.path.basename(source)
	else:
		# Get the first (and only) slide from directory
		slides = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
		if not slides:
			raise ValueError(f"No slide files found in {source}")
		if len(slides) > 1:
			print(f"Warning: Multiple slides found, processing only the first one: {slides[0]}")
		
		slide_name = slides[0]
		slide_path = os.path.join(source, slide_name)
	
	print(f'Processing slide: {slide_name}')
	
	# Initialize WSI object
	WSI_object = WholeSlideImage(slide_path)
	
	# Process parameters - use copies to avoid modifying originals
	current_seg_params = seg_params.copy()
	current_filter_params = filter_params.copy()
	current_patch_params = {}
	
	# Determine best levels if not specified
	def get_best_level():
		if len(WSI_object.level_dim) == 1:
			return 0
		else:
			wsi = WSI_object.getOpenSlide()
			return wsi.get_best_level_for_downsample(64)
	
	# Set segmentation level
	if current_seg_params['seg_level'] < 0:
		current_seg_params['seg_level'] = get_best_level()
	
	# Process keep_ids and exclude_ids
	def process_ids(ids_str):
		if ids_str == 'none' or not ids_str:
			return []
		return np.array(ids_str.split(',')).astype(int)
	
	current_seg_params['keep_ids'] = process_ids(str(current_seg_params['keep_ids']))
	current_seg_params['exclude_ids'] = process_ids(str(current_seg_params['exclude_ids']))
	
	# Check if slide dimensions are reasonable for segmentation
	w, h = WSI_object.level_dim[current_seg_params['seg_level']]
	print(f"Slide dimensions at seg_level {current_seg_params['seg_level']}: {w} x {h} = {w*h:,.0f} pixels")
	
	if w * h > 1e8:
		raise ValueError(f'Slide dimensions {w} x {h} are too large for segmentation (> 100M pixels)')
	

	WSI_object, _ = segment(WSI_object, current_seg_params, current_filter_params)
	
	current_patch_params.update({
		'patch_level': patch_level, 
		'patch_size': patch_size, 
		'step_size': step_size,
		'save_path': patch_save_dir
	})
	file_path = patching(WSI_object=WSI_object, **current_patch_params)
	
	# Return slide information as a dictionary for direct use
	slide_info = {
		'slide_id': slide_name,
		'slide_path': slide_path,
		'file_path': file_path,
		'process': 0,  # 0 means processed
		'status': 'processed',
		'seg_level': current_seg_params['seg_level'],
	}
	
	print("Processing completed successfully")
 
	del WSI_object  # Clean up WSI object to free memory
	del current_seg_params, current_filter_params, current_patch_params  # Clean up parameters
 
	return slide_info

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params}

	print(parameters)

	result = seg_and_patch(**directories, **parameters,
						   patch_size=args.patch_size, step_size=args.step_size, 
						   patch_level=args.patch_level,
						   best_level=args.best_level)
	
	print(f"Processing completed. Result: {result}")
