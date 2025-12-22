#!/usr/bin/env python3
"""
Script to merge federated learning splits into centralized splits.

This script takes split files from a federated setup (3 clients, 5-fold CV) and 
merges them into centralized splits where training, validation, and test sets 
from all clients are combined for each fold.

Input structure: chimera_3_5_0.1_1/splits_{client_id}_{fold_id}.csv
Output structure: chimera_3_5_0.1_1_merged/centralized_splits_{fold_id}.csv
"""

import os
import pandas as pd
from pathlib import Path


def merge_splits_to_centralized(input_dir, output_dir, num_clients=3, num_folds=5):
    """
    Merge federated splits into centralized splits.
    
    Args:
        input_dir (str): Directory containing the federated split files
        output_dir (str): Directory where centralized splits will be saved
        num_clients (int): Number of clients (default: 3)
        num_folds (int): Number of folds (default: 5)
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Merging splits from {input_dir} to {output_dir}")
    print(f"Configuration: {num_clients} clients, {num_folds} folds")
    
    # Process each fold
    for fold_id in range(num_folds):
        print(f"\nProcessing fold {fold_id}...")
        
        # Initialize lists to store data from all clients for this fold
        all_train = []
        all_val = []
        all_test = []
        
        # Load data from all clients for this fold
        for client_id in range(num_clients):
            split_file = os.path.join(input_dir, f"splits_{client_id}_{fold_id}.csv")
            
            if not os.path.exists(split_file):
                print(f"Warning: File {split_file} not found. Skipping client {client_id} for fold {fold_id}")
                continue
                
            print(f"  Loading {split_file}")
            
            # Read the split file
            df = pd.read_csv(split_file, index_col=0)
            
            # Extract non-empty values from each split
            train_samples = df['train'].dropna().tolist()
            val_samples = df['val'].dropna().tolist()
            test_samples = df['test'].dropna().tolist()
            
            # Add to combined lists
            all_train.extend(train_samples)
            all_val.extend(val_samples)
            all_test.extend(test_samples)
            
            print(f"    Client {client_id}: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        # Create centralized split DataFrame
        max_length = max(len(all_train), len(all_val), len(all_test))
        
        # Pad lists to same length with empty strings
        all_train.extend([''] * (max_length - len(all_train)))
        all_val.extend([''] * (max_length - len(all_val)))
        all_test.extend([''] * (max_length - len(all_test)))
        
        # Create DataFrame
        centralized_df = pd.DataFrame({
            'train': all_train,
            'val': all_val,
            'test': all_test
        })
        
        # Save centralized split
        output_file = os.path.join(output_dir, f"splits_0_{fold_id}.csv")
        centralized_df.to_csv(output_file)
        
        print(f"  Saved centralized fold {fold_id}: {len(all_train)} train, {len(all_val)} val, {len(all_test)} test samples")
        print(f"  Output: {output_file}")
    
    print(f"\nMerging completed! Centralized splits saved in {output_dir}")


def main():
    # Define paths
    base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/splits"
    for i in range(2, 6):
        input_dir = os.path.join(base_dir, f"chimera_3_5_0.1_{i}")
        output_dir = os.path.join(base_dir, f"chimera_3_5_0.1_{i}_merged")
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist!")
            return
        
        # List files in input directory
        print("Files found in input directory:")
        for file in sorted(os.listdir(input_dir)):
            if file.endswith('.csv'):
                print(f"  {file}")
        
        # Merge splits
        merge_splits_to_centralized(input_dir, output_dir, num_clients=3, num_folds=5)
        
        # Verify output
        print(f"\nVerification - Files created in output directory:")
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                if file.endswith('.csv'):
                    file_path = os.path.join(output_dir, file)
                    df = pd.read_csv(file_path, index_col=0)
                    train_count = (df['train'] != '').sum()
                    val_count = (df['val'] != '').sum()
                    test_count = (df['test'] != '').sum()
                    print(f"  {file}: {train_count} train, {val_count} val, {test_count} test")


if __name__ == "__main__":
    main()