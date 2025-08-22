import numpy as np
from typing import List, Tuple
from dataset.dataset_iterator import root_iter, id_to_filename
from sklearn.model_selection import StratifiedKFold
import json
import os

def create_kfold_splits(k: int = 10, test_p = 0.1, balance: bool = True, seed: int = 1, verbose = False, as_filename = False):
    """
    Create k-fold cross-validation splits of the dataset.
    
    Args:
        dataset (ChimeraCDDataset): The dataset to split
        k (int): Number of folds
        balance (bool): Whether to balance classes in each fold
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of k tuples, each containing (train_dataset, val_dataset)
    """
    
    np.random.seed(seed)
    
    
    root_iterator = root_iter(clinical_only=True)
        
    label_to_int = {
        "BRS1": 0,
        "BRS2": 1,
        "BRS3": 2,
    }
    ids = []
    labels = {}
    
    for id, data in root_iterator:
        ids.append(id)
        cd_path = data["cd"]
        assert os.path.exists(cd_path), f"CD file {cd_path} does not exist"
        
        with open(cd_path, "r") as input_file:
            cd = json.load(input_file)
            labels[id] = label_to_int[cd["BRS"]]
    
    
    if balance:
        # Exclude test set from k-fold
        labels = labels
        ids_per_label = {label: [i for i in ids if labels[i] == label] for label in np.unique(list(labels.values()))}
        
        train_val_ids = []
        test_ids = []
        
        for l in list(ids_per_label.keys()):
            
            test_size = int(len(ids_per_label[l]) * test_p)
            
            ids_of_label = ids_per_label[l]
            np.random.shuffle(ids_of_label)
            if test_size == 0:
                train_val_ids.extend(ids_of_label)
            else:
                train_val_ids.extend(ids_of_label[:-test_size])
                test_ids.extend(ids_of_label[-test_size:])
        assert len(np.intersect1d(train_val_ids, test_ids)) == 0
        
        # Use StratifiedKFold to ensure class balance
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        train_val_labels = [labels[i] for i in train_val_ids]
        for i, (train_index, val_index) in enumerate(skf.split(train_val_ids, train_val_labels)):
            train_ids = [train_val_ids[j] for j in train_index]
            val_ids = [train_val_ids[j] for j in val_index]

            assert len(np.intersect1d(train_ids, val_ids)) == 0
            assert len(np.intersect1d(train_ids, test_ids)) == 0
            assert len(np.intersect1d(val_ids, test_ids)) == 0
            
            # Print statistics
            if verbose:
                print(f"\nFold {i+1}/{k}:")
                print(f"Training: {len(train_ids)} samples, label distribution: {np.unique([labels[i] for i in train_ids], return_counts=True)}")
                print("Ids:", train_ids)
                print(f"Validation: {len(val_ids)} samples, label distribution: {np.unique([labels[i] for i in val_ids], return_counts=True)}")
                print("Ids:", val_ids)
                print(f"Testing: {len(test_ids)} samples, label distribution: {np.unique([labels[i] for i in test_ids], return_counts=True)}")
                print("Ids:", test_ids)
            if as_filename:
                train_ids_as_filename = [id_to_filename(i)+"_HE" for i in train_ids]
                val_ids_as_filename = [id_to_filename(i)+"_HE" for i in val_ids]
                test_ids_as_filename = [id_to_filename(i)+"_HE" for i in test_ids]
                yield train_ids_as_filename, val_ids_as_filename, test_ids_as_filename
            else:
                yield train_ids, val_ids, test_ids

def create_csv_split(folds: List[Tuple[List[int], List[int], List[int]]], name:str, outputs_dir: str):
    """    Create a CSV file with the split information, that looks like this:
    ,train,val,test
    0,2A_005_HE,2A_009_HE,2A_127_HE
    1,2A_006_HE,2A_149_HE,2A_154_HE
    ...
    """
    import os
    import pandas as pd
    
    outputs_dir = os.path.join(outputs_dir, name)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    split_names = ['train', 'val', 'test']
    
    
    for f, fold in enumerate(folds):
        split_data = {name: [] for name in split_names}
        for i, split_ids in enumerate(fold):
            split_data[split_names[i]].extend(split_ids)
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_data.items()]))
        df.to_csv(f"{outputs_dir}/splits_{f}.csv")
 
if __name__ == "__main__":
    k = 10
    test_p = 0.1
    balance = True
    seed = 5523101
    output_dir = "splits/"
    
    splits_generator = create_kfold_splits(k=k, test_p=test_p, balance=balance, seed=seed, verbose=True, as_filename=True)
    
    splits = list(splits_generator)
    create_csv_split(splits, name=f"chimera_1_{k}_{test_p}", outputs_dir=output_dir)
