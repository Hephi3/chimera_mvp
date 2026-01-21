import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 file_path,
                 img_transforms=None):
        """
        Args:
                file_path (string): Path to the .h5 file containing patched data.
                roi_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.roi_transforms = img_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        with h5py.File(self.file_path, "r") as hdf5_file:
            dset = hdf5_file['imgs']
            for name, value in dset.attrs.items():
                print(name, value)

        print('transformations:', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 img_transforms=None,
                 use_stain_norm=True,
                 use_mask=True,
                 augmentation=None,
                 mask_path=None,
                 target_image_path=None
                 ):
        """
        Args:
                file_path (string): Path to the .h5 file containing patched data.
                img_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path
        self.use_stain_norm = use_stain_norm
        self.use_mask = use_mask
        self.mask_wsi = None
        self.augmentation = augmentation
        self.mask_path = mask_path
        if mask_path is None:
            use_mask = False
            self.use_mask = False
        if use_mask:
            self._setup_mask()
            

        # MY CODE: STAIN NORMALIZATION
        if use_stain_norm:
            import torchstain
            import torch
            # Load target image
            target = np.array(Image.open(target_image_path))

            # Create normalizer
            self.normalizer = torchstain.normalizers.MacenkoNormalizer(
                backend='torch')
            self.normalizer.fit(torch.from_numpy(
                target).permute(2, 0, 1).float())

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)

        self.summary()

    def _setup_mask(self):
        """Set up the mask WSI if available"""
        import os
        
        # Use provided mask path if available, otherwise use default logic
        mask_path = self.mask_path

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Warning: Mask file {mask_path} does not exist. Proceeding without masking.")

        import tifffile
        print(f"Loading mask from {mask_path}...")
        with tifffile.TiffFile(mask_path) as tif:
            self.mask_wsi = tif.asarray()
            self.mask_wsi = Image.fromarray(self.mask_wsi)
            self.mask_wsi = self.mask_wsi.convert('L')


    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(
            coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        
        if self.augmentation is not None:
            img = self.augmentation(img)
        
        if self.use_stain_norm:
            import torch
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            try:
                normalized, _, _ = self.normalizer.normalize(img_t)
            except Exception as e:
                # Return a white image if normalization fails
                img = torch.ones_like(img_t)
                return {'img': img, 'coord': coord}
                
            img = normalized.byte().cpu().numpy()
            img = Image.fromarray(img)

        if self.use_mask and self.mask_wsi is not None:
            # Get the dimensions of both the WSI and mask at level 0
            wsi_w, wsi_h = self.wsi.dimensions
            mask_w, mask_h = self.mask_wsi.size

            # Calculate scaling factors between WSI and mask
            scale_x = mask_w / wsi_w
            scale_y = mask_h / wsi_h

            # Scale the coordinates to match the mask dimensions
            mask_x = int(coord[0] * scale_x)
            mask_y = int(coord[1] * scale_y)

            # Calculate mask patch size (accounting for scaling)
            mask_patch_size = int(self.patch_size * scale_x)

            mask_region = self.mask_wsi.crop(
                (mask_x, mask_y, mask_x + mask_patch_size, mask_y + mask_patch_size))

            # Resize to match the patch size if needed
            if mask_region.size != (self.patch_size, self.patch_size):
                mask_region = mask_region.resize(
                    (self.patch_size, self.patch_size), Image.NEAREST)

            # Convert to binary mask (0 or 1)
            mask_array = np.array(mask_region)
            mask_array = (mask_array > 0).astype(np.float32)
            
            
            img_stained = img
            # Apply mask to the image
            img_array = np.array(img)
            masked_array = (
                img_array * mask_array[:, :, np.newaxis]).astype(np.uint8)
            img = Image.fromarray(masked_array)
        
        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}


class Dataset_All_Bags(Dataset):

    def __init__(self, csv):
        self.df = csv

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
