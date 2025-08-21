import os
from torchvision.transforms import v2

ROOT_DIR = ROOT_DIR = "/local/scratch/chimera/task2new/data"

def id_to_filename(id, he = False):
    filename = f"2{'A' if int(id)<200 else 'B'}_{id}"
    if he:
        filename += "_HE"
    return filename

def root_iter(root_dir=ROOT_DIR, page=None, clinical_only = False, tumor_masks = False):
    """Returns an iterator over patient IDs and their associated files."""
    
    def patient_iterator():
        for patient_dir in sorted(os.listdir(root_dir)):
            id = patient_dir.split("_")[-1]
            # Skip problematic files
            if id in ["082", "086", "025"]:
                continue
            
            patient_path = os.path.join(root_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
                
            # Check required files
            if not clinical_only:
                if page is not None:
                    hist_path = os.path.join(patient_path, f"{patient_dir}_HE_{page}.tif")
                    if not os.path.exists(hist_path):
                        # print(f"Warning: {hist_path} does not exist. Skipping patient {id}.")
                        continue
                else:
                    hist_path = os.path.join(patient_path, f"{patient_dir}_HE.tif")
                    if not os.path.exists(hist_path):
                        print(f"Warning: {hist_path} does not exist. Skipping patient {id}.")
                        continue
                
                hist_mask_path = os.path.join(patient_path, f"{patient_dir}_HE_mask.tif")
            cd_path = os.path.join(patient_path, f"{patient_dir}_CD.json")
            
            # if not (os.path.exists(hist_path) and os.path.exists(hist_mask_path) and os.path.exists(cd_path)):
            #     continue
            result = {
                "cd": cd_path
            }
            if not clinical_only:
                result["hist"] = hist_path
                result["hist_mask"] = hist_mask_path
            
            if tumor_masks:
                # from rootdir everything but the last part of the path
                tumor_mask_path = os.path.join("/local/scratch/chimera/task2new/", "RNA_masks",f"{patient_dir[1:]}.tif")
                if os.path.exists(tumor_mask_path):
                    result["tumor_mask"] = tumor_mask_path
                else:
                    print(f"Warning: Tumor mask {tumor_mask_path} does not exist!")
                    result["tumor_mask"] = None

            yield id, result
    
    return patient_iterator()
