#!/usr/bin/env python3
"""
Script to swap test IDs between consecutive stages in split files.
For each client and fold, swaps test IDs between stage 0 and stage 1.
"""

import os
import pandas as pd
from pathlib import Path


def swap_test_ids_between_stages(base_dir, folder_pattern, splitnr_list):
    """
    Swap test IDs between stage 0 and stage 1 for all splits in specified folders.
    
    Args:
        base_dir: Base directory containing the split folders
        folder_pattern: Pattern for folder names (e.g., "chimera_3_5_2_0.2_0.5_0.5_{}_reversed")
        splitnr_list: List of split numbers to process
    """
    
    for splitnr in splitnr_list:
        folder_name = folder_pattern.format(splitnr)
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")
            continue
        
        print(f"\nProcessing folder: {folder_name}")
        
        # Get all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        # Group files by client and fold
        splits_by_client_fold = {}
        for csv_file in csv_files:
            # Parse filename: splits_{clientnr}_{foldnr}_{stagenr}.csv
            parts = csv_file.replace('.csv', '').split('_')
            if len(parts) == 4 and parts[0] == 'splits':
                clientnr = parts[1]
                foldnr = parts[2]
                stagenr = parts[3]
                key = (clientnr, foldnr)
                
                if key not in splits_by_client_fold:
                    splits_by_client_fold[key] = {}
                splits_by_client_fold[key][stagenr] = csv_file
        
        # Swap test IDs between stage 0 and stage 1
        swaps_made = 0
        for (clientnr, foldnr), stages in splits_by_client_fold.items():
            if '0' in stages and '1' in stages:
                file_stage0 = os.path.join(folder_path, stages['0'])
                file_stage1 = os.path.join(folder_path, stages['1'])
                
                # Read both CSV files
                df_stage0 = pd.read_csv(file_stage0, index_col=0)
                df_stage1 = pd.read_csv(file_stage1, index_col=0)
                
                # Swap the test columns
                test_stage0 = df_stage0['test'].copy()
                test_stage1 = df_stage1['test'].copy()
                
                df_stage0['test'] = test_stage1
                df_stage1['test'] = test_stage0
                
                # Save the modified files
                df_stage0.to_csv(file_stage0)
                df_stage1.to_csv(file_stage1)
                
                swaps_made += 1
                print(f"  Swapped test IDs for client {clientnr}, fold {foldnr}")
        
        print(f"Total swaps made in {folder_name}: {swaps_made}")


def main():
    # Configuration
    base_dir = "splits"
    folder_pattern = "chimera_3_5_2_0.2_0.5_0.5_{}_reversed"
    splitnr_list = [1, 2, 3, 4, 5]
    
    # Get absolute path
    script_dir = Path(__file__).parent
    base_dir_path = script_dir / base_dir
    
    print(f"Base directory: {base_dir_path}")
    print(f"Split numbers to process: {splitnr_list}")
    print("=" * 60)
    
    swap_test_ids_between_stages(str(base_dir_path), folder_pattern, splitnr_list)
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
