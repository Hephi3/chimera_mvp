import numpy as np
from typing import List, Tuple
from dataset.dataset_iterator import root_iter, id_to_filename
from sklearn.model_selection import StratifiedKFold
import json
import os

def create_k_clients_splits(num_clients: int, test_p: float = 0.1, val_p:float = 0.2, balance: bool = True, seed: int = 1, verbose = False, as_filename = False):
    """
    Create num_clients training splits, holding out a test set.
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
        # Exclude test set from client train sets
        labels = labels
        ids_per_label = {label: [i for i in ids if labels[i] == label] for label in np.unique(list(labels.values()))}
        # -> {0: [id1, id2, ...], 1: [id3, id4, ...], 2: [id5, id6, ...]}
        
        train_val_ids = [] # -> all ids that are not in the test set
        test_ids = [] # -> all ids that are in the test set
        
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
        
        if num_clients == 1:
            # Just create a single split
            train_ids = train_val_ids
            val_size = int(len(train_ids) * val_p)
            if val_size == 0:
                val_ids = []
            else:
                val_ids = np.random.choice(train_ids, size=val_size, replace=False).tolist()
                train_ids = [i for i in train_ids if i not in val_ids]
            
            # Print statistics
            if verbose:
                print(f"\Client 1/{num_clients}:")
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
            return
            
        
        
        # Use StratifiedKFold to ensure class balance
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
        train_val_labels = [labels[i] for i in train_val_ids]        
        
        
        for i, (others_data_idx, train_data_idx) in enumerate(skf.split(train_val_ids, train_val_labels)):
            train_ids = [train_val_ids[j] for j in train_data_idx]

            assert len(np.intersect1d(train_ids, test_ids)) == 0
            
            # Divide train_ids into train and val with stratified sampling
            val_size = int(len(train_ids) * val_p)
            if val_size == 0:
                val_ids = []
            else:
                # Use stratified sampling for validation split to maintain class balance
                train_labels_for_split = [labels[id] for id in train_ids]
                
                # Create a simple stratified split for train/val
                val_skf = StratifiedKFold(n_splits=int(1/val_p), shuffle=True, random_state=seed+i)
                train_split_idx, val_split_idx = next(val_skf.split(train_ids, train_labels_for_split))
                
                val_ids = [train_ids[j] for j in val_split_idx]
                train_ids = [train_ids[j] for j in train_split_idx]
            
            # Print statistics
            if verbose:
                print(f"\Client {i+1}/{num_clients}:")
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

def create_csv_split(clients: List[Tuple[List[int], List[int], List[int]]], name:str, outputs_dir: str):
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
    
    # Assert that there is no overlap between the train sets from different clients
    seen_ids = set()
    for f, client in enumerate(clients):
        split_data = {name: [] for name in split_names}
        for i, split_ids in enumerate(client):
            split_data[split_names[i]].extend(split_ids)
        
        for id in split_data['train']:
            assert id not in seen_ids, f"ID {id} is in the training set of multiple clients!"
            seen_ids.add(id)
        for id in split_data['val']:
            assert id not in seen_ids, f"ID {id} is in the validation set of multiple clients!"
            seen_ids.add(id)
        
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_data.items()]))
        df.to_csv(f"{outputs_dir}/splits_{f}.csv")
 
if __name__ == "__main__":
    num_clients = 1
    test_p = 0.1
    val_p = 0.2
    balance = True
    seed = 1
    output_dir = "splits/"

    splits_generator = create_k_clients_splits(num_clients=num_clients, test_p=test_p, val_p=val_p, balance=balance, seed=seed, verbose=True, as_filename=True)
    
    splits = list(splits_generator)
    create_csv_split(splits, name=f"chimera_{num_clients}_{test_p}", outputs_dir=output_dir)
