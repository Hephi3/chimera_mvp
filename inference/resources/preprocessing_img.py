

import timm
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import List, Tuple
import os
import tifffile
# from stainNorm_Macenko import Normalizer

def open_image_and_mask(image_path, mask_path, page) -> Tuple[Image.Image, Image.Image]:
    
    assert os.path.exists(image_path) and os.path.exists(mask_path), f"Image file {image_path} or mask file {mask_path} does not exist"
    
    with tifffile.TiffFile(image_path) as tif:
        if page >= len(tif.pages):
            print(f"Page {page} is not valid. Page should be < {len(tif.pages)}")
            return None
        arr = tif.pages[page].asarray()
        img = Image.fromarray(arr)
    
    with tifffile.TiffFile(mask_path) as tif:
        if page >= len(tif.pages):
            print(f"Page {page} is not valid. Page should be < {len(tif.pages)}")
            return None
        arr = tif.pages[page].asarray()
        mask = Image.fromarray(arr)
    # if id == "025":
    #     mask = create_foreground_mask(img, plt_show=False, plt_scale=False)
    if img.size != mask.size:
        mask = mask.resize(img.size)
    return img, mask

def extract_contour_patches(image: Image.Image, mask: Image.Image, patch_size: int, overlap_ratio: float, threshold: float) -> List[np.ndarray]:
    patches = []

    step = int(patch_size * (1 - overlap_ratio))  # Calculate step size based on overlap ratio
    shape = np.array(image).shape
    for y in range(0, shape[0], step):
        for x in range(0, shape[1], step):
            coordinates = (x, y, x + patch_size, y + patch_size)
            
            img_patch = np.array(image.crop(coordinates))
            mask_patch = np.array(mask.crop(coordinates))
            
            # Handling patches at the edges
            if img_patch.shape[0] < patch_size or img_patch.shape[1] < patch_size:
                img_patch = np.pad(img_patch, ((0, patch_size - img_patch.shape[0]),
                                               (0, patch_size - img_patch.shape[1]),
                                               (0, 0)), mode='constant', constant_values=255)
                mask_patch = np.pad(mask_patch, ((0, patch_size - mask_patch.shape[0]),
                                                 (0, patch_size - mask_patch.shape[1])),
                                    mode='constant')
            
            # If patch contains at least threshold% of the foreground
            if np.sum(mask_patch) > threshold * patch_size ** 2:
                patches.append(img_patch)
            
    del image, mask  # Free memory

    return patches

# def stain_normalization(patches):    
#     target_patch_path = "/local/scratch/phempel/chimera/features_MMFL/page_1/224/0.7/039/1290.png"
#     target_patch = Image.open(target_patch_path)
#     target_patch = np.array(target_patch)
#     normalizer = Normalizer()
#     print("Fitting normalizer...")
#     normalizer.fit(target_patch)
    
#     normalized_patches = []
    
#     for patch in patches, desc="Loading and normalizing patches":
#         normalized_patches.append(normalizer.transform(np.array(patch)))

# def stain_normalization_img(image):    
#     target_patch_path = "/local/scratch/phempel/chimera/features_MMFL/page_1/224/0.7/039/1290.png"
#     target_patch = Image.open(target_patch_path)
#     target_patch = np.array(target_patch)
#     normalizer = Normalizer()
#     print("Fitting normalizer...")
#     normalizer.fit(target_patch)
#     print("Normalizing image...")
    
#     return normalizer.transform(np.array(image))










def get_encoder(uni_model_path):
    
    print(f"Loading model from: {uni_model_path}")
    if not os.path.exists(uni_model_path):
        raise FileNotFoundError(f"Model file not found: {uni_model_path}")
    
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    
    print("Creating model...")
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    
    print("Loading model weights...")
    model.load_state_dict(torch.load(uni_model_path, map_location="cpu"), strict=True)
    print("Model loaded successfully!")
    
    img_transforms = get_eval_transforms(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225],
                                         target_img_size = 224)
    
    return model, img_transforms
    
def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms

def extract_features(img, uni_model_path, device):
    model, img_transforms = get_encoder(uni_model_path)
    
    _ = model.eval()
    model = model.to(device)
    
    img = img_transforms(img)
    img = img.unsqueeze(0)  # Add batch dimension
    img = img.to(device)
    with torch.no_grad():
        features = model(img)
    features = features.squeeze(0)  # Remove batch dimension
    features = features.cpu().numpy()  # Convert to numpy array
    return features

def extract_features_batchwise(patches, uni_model_path, device, batch_size=32):
    model, img_transforms = get_encoder(uni_model_path)
    model.eval()
    model = model.to(device)
    
    all_features = []
    
    # Process patches in batches
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        
        # Transform all patches in the batch
        batch_tensors = []
        for patch in batch_patches:
            if isinstance(patch, np.ndarray):
                patch = Image.fromarray(patch)
            tensor = img_transforms(patch)
            batch_tensors.append(tensor)
        
        # Stack into batch tensor [batch_size, 3, 224, 224]
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            batch_features = model(batch_tensor)
            batch_features = batch_features.cpu().numpy()
            all_features.append(batch_features)
    
    # Concatenate all batches
    return np.concatenate(all_features, axis=0)


def preproc_img_file(image_path, mask_path, uni_model_path):
    print("Opening image and mask...")
    image, mask = open_image_and_mask(image_path, mask_path, 2)
    print("Image and mask opened successfully.")
    
    # image = stain_normalization_img(image)
    
    print("Extracting contour patches...")
    patches = extract_contour_patches(image, mask, 224, 0, threshold= 0.7)
    del image, mask  # Free memory
    print(f"Extracted {len(patches)} patches.")
    
    if len(patches) == 0:
        print("No patches extracted! Returning empty features.")
        return np.array([])
    
    # print("Normalizing patches...")
    # patches = stain_normalization(patches)
    # print(f"Normalized patches count: {len(patches)}")
    
    print("Extracting features from patches...")
    features = extract_features_batchwise(patches, uni_model_path, batch_size=32)
    print(f"Extracted features shape: {features.shape}")
    return features

if __name__ == '__main__':
    preproc_img_file("/local/scratch/chimera/task2new/data/2B_362/2B_362_HE.tif", "/local/scratch/chimera/task2new/data/2B_362/2B_362_HE_mask.tif")
    # # img_path = 'path_to_your_image.jpg'  # Replace with your image path
    # # img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
    # patch_path = '/local/scratch/phempel/chimera/features_MMFL/page_1/224/0.7/001/4.png'
    # patch = Image.open(patch_path).convert('RGB')  # Ensure image is in
    # features = extract_features(patch)
    # print(features.shape)  # Should print the shape of the extracted features
    # print(features)  # Print the extracted features
    
    
    
    
    
    
    
    
    
    
    
    
    
    
