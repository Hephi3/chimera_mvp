import os
import json
from pathlib import Path
import uuid

def create_input_structure():
    """
    Creates the required input structure with symbolic links to the original data
    Creates all 182 cases in separate interface_{idx} folders
    """
    # Source directory containing the 182 folders
    source_dir = Path("/local/scratch/chimera/task2new/data")
    
    # Target directory structure
    base_input_dir = Path("/gris/gris-f/homelv/phempel/masterthesis/test/input")
    
    # Process each folder in the source directory
    folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} folders to process")
    
    successful_cases = 0
    
    for idx, folder in enumerate(folders):
        folder_name = folder.name
        print(f"Processing folder {idx}: {folder_name}")
        
        # Find files in the folder
        json_files = list(folder.glob("*.json"))
        he_files = list(folder.glob("*HE.tif"))
        mask_files = list(folder.glob("*HE_mask.tif"))
        
        if not json_files or not he_files or not mask_files:
            print(f"Warning: Missing files in {folder_name}")
            print(f"  JSON files: {len(json_files)}")
            print(f"  HE files: {len(he_files)}")
            print(f"  Mask files: {len(mask_files)}")
            continue
        
        # Create interface directory for this case
        interface_dir = base_input_dir / f"interface_{idx}"
        tissue_mask_dir = interface_dir / "images" / "tissue-mask"
        wsi_dir = interface_dir / "images" / "bladder-cancer-tissue-biopsy-wsi"
        
        tissue_mask_dir.mkdir(parents=True, exist_ok=True)
        wsi_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the folder name as the base UUID
        base_uuid = folder_name
        
        # Create symbolic links for HE image
        he_source = he_files[0]
        he_target = wsi_dir / f"{base_uuid}.tif"
        if he_target.exists():
            he_target.unlink()  # Remove existing link
        he_target.symlink_to(he_source.absolute())
        print(f"  Created WSI link: {he_target} -> {he_source}")
        
        # Create symbolic links for mask
        mask_source = mask_files[0]
        mask_target = tissue_mask_dir / f"{base_uuid}.tif"
        if mask_target.exists():
            mask_target.unlink()  # Remove existing link
        mask_target.symlink_to(mask_source.absolute())
        print(f"  Created mask link: {mask_target} -> {mask_source}")
        
        # Create symbolic link for clinical data
        json_source = json_files[0]
        clinical_data_file = interface_dir / "chimera-clinical-data-of-bladder-cancer-patients.json"
        if clinical_data_file.exists():
            clinical_data_file.unlink()  # Remove existing link
        clinical_data_file.symlink_to(json_source.absolute())
        print(f"  Created clinical data link: {clinical_data_file} -> {json_source}")
        
        # Create inputs.json file for this interface
        inputs_json = {
            "inputs": [
                {
                    "interface": {
                        "slug": "bladder-cancer-tissue-biopsy-wsi",
                        "kind": "Image"
                    }
                },
                {
                    "interface": {
                        "slug": "tissue-mask", 
                        "kind": "Segmentation"
                    }
                },
                {
                    "interface": {
                        "slug": "chimera-clinical-data-of-bladder-cancer-patients",
                        "kind": "Anything"
                    }
                }
            ]
        }
        
        inputs_file = interface_dir / "inputs.json"
        with open(inputs_file, 'w') as f:
            json.dump(inputs_json, f, indent=2)
        print(f"  Created inputs.json: {inputs_file}")
        
        successful_cases += 1
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Processed {len(folders)} folders")
    print(f"Successfully created {successful_cases} interface directories")
    print(f"Input structure created at: {base_input_dir}")
    print(f"Interface directories: interface_0 to interface_{successful_cases-1}")
    print(f"\nTo test a specific case, modify do_test_run.sh to use:")
    print(f"  run_docker_forward_pass \"interface_X\"")
    print(f"where X is the interface number (0 to {successful_cases-1})")


def create_single_case_structure(case_id=None):
    """
    Creates input structure for a single case in interface_0 (useful for testing with do_test_run.sh)
    """
    source_dir = Path("/local/scratch/chimera/task2new/data")
    # Create structure in interface_0 for compatibility with do_test_run.sh
    base_input_dir = Path("/gris/gris-f/homelv/phempel/masterthesis/test/input/interface_0")
    
    # If no case_id specified, use the first available folder
    if case_id is None:
        folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
        if not folders:
            print("No folders found in source directory")
            return
        case_folder = folders[0]
        case_id = case_folder.name
    else:
        case_folder = source_dir / case_id
        if not case_folder.exists():
            print(f"Case folder {case_id} not found")
            return
    
    print(f"Creating single case structure for: {case_id} in interface_0")
    
    # Create target directories
    tissue_mask_dir = base_input_dir / "images" / "tissue-mask"
    wsi_dir = base_input_dir / "images" / "bladder-cancer-tissue-biopsy-wsi"
    
    tissue_mask_dir.mkdir(parents=True, exist_ok=True)
    wsi_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    json_files = list(case_folder.glob("*.json"))
    he_files = list(case_folder.glob("*HE.tif"))
    mask_files = list(case_folder.glob("*HE_mask.tif"))
    
    if not json_files or not he_files or not mask_files:
        print(f"Error: Missing files in {case_id}")
        return
    
    # Create links (remove existing ones first)
    he_target = wsi_dir / f"{case_id}.tif"
    mask_target = tissue_mask_dir / f"{case_id}.tif"
    clinical_target = base_input_dir / "chimera-clinical-data-of-bladder-cancer-patients.json"
    
    if he_target.exists():
        he_target.unlink()
    if mask_target.exists():
        mask_target.unlink()
    
    he_target.symlink_to(he_files[0].absolute())
    mask_target.symlink_to(mask_files[0].absolute())
    
    # Create symbolic link for clinical data
    json_source = json_files[0]
    clinical_target = base_input_dir / "chimera-clinical-data-of-bladder-cancer-patients.json"
    if clinical_target.exists():
        clinical_target.unlink()  # Remove existing link
    clinical_target.symlink_to(json_source.absolute())
    print(f"  Created clinical data link: {clinical_target} -> {json_source}")
    
    # Create inputs.json
    inputs_json = {
        "inputs": [
            {
                "interface": {
                    "slug": "bladder-cancer-tissue-biopsy-wsi",
                    "kind": "Image"
                }
            },
            {
                "interface": {
                    "slug": "tissue-mask",
                    "kind": "Segmentation"
                }
            },
            {
                "interface": {
                    "slug": "chimera-clinical-data-of-bladder-cancer-patients",
                    "kind": "Anything"
                }
            }
        ]
    }
    
    inputs_file = base_input_dir / "inputs.json"
    with open(inputs_file, 'w') as f:
        json.dump(inputs_json, f, indent=2)
    
    print(f"Single case structure created for {case_id}")
    print(f"WSI: {he_target}")
    print(f"Mask: {mask_target}")
    print(f"Clinical data: {clinical_target}")
    print(f"Ready for: ./do_test_run.sh")


def create_test_all_script():
    """
    Creates a script to test all interface directories
    """
    base_input_dir = Path("/gris/gris-f/homelv/phempel/masterthesis/test/input")
    
    # Find all interface directories
    interface_dirs = sorted([d for d in base_input_dir.iterdir() if d.is_dir() and d.name.startswith("interface_")])
    
    if not interface_dirs:
        print("No interface directories found. Run the main script first.")
        return
    
    # Create test script
    script_path = Path("/gris/gris-f/homelv/phempel/masterthesis/test/test_all_cases.sh")
    
    script_content = f"""#!/usr/bin/env bash

# Script to test all {len(interface_dirs)} cases
# Generated automatically by add_input.py

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
CHIMERA_DIR="$SCRIPT_DIR/../chimera"

cd "$CHIMERA_DIR"

echo "Testing {len(interface_dirs)} cases..."

# Test each interface
"""
    
    for i, interface_dir in enumerate(interface_dirs):
        interface_name = interface_dir.name
        script_content += f"""
echo "=== Testing {interface_name} ({i+1}/{len(interface_dirs)}) ==="
# Modify do_test_run.sh to use {interface_name}
sed -i 's/run_docker_forward_pass "interface_[0-9]*"/run_docker_forward_pass "{interface_name}"/g' do_test_run.sh
./do_test_run.sh
echo "Completed {interface_name}"
echo
"""
    
    script_content += """
echo "All tests completed!"
echo "Results are in: $SCRIPT_DIR/output/"
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    script_path.chmod(0o755)
    
    print(f"Created test script: {script_path}")
    print(f"To test all {len(interface_dirs)} cases, run:")
    print(f"  {script_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            case_id = sys.argv[2] if len(sys.argv) > 2 else None
            create_single_case_structure(case_id)
        elif sys.argv[1] == "test-script":
            create_test_all_script()
        else:
            print("Usage:")
            print("  python add_input.py                    # Create all cases in interface_X folders")
            print("  python add_input.py single [case_id]   # Create single case in interface_0")
            print("  python add_input.py test-script        # Create script to test all cases")
    else:
        create_input_structure()