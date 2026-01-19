import numpy as np
from typing import List, Tuple
from dataset.dataset_iterator import root_iter, id_to_filename
from sklearn.model_selection import StratifiedKFold, train_test_split
import json
import os

def create_k_clients_cross_fold_splits_stages(num_clients: int, test_p: float = 0.1, folds:int = 1, stages:int = 1, balance: bool = True, seed: int = 1, verbose = False, as_filename = False):
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
        skf_stages = StratifiedKFold(n_splits=stages, shuffle=True, random_state=seed)
        label_list = [labels[id] for id in ids]

        for s, (others_data_idx, stage_data_idx) in enumerate(skf_stages.split(ids, label_list)):
            stage_ids = [ids[j] for j in stage_data_idx]
            stage_labels = [labels[j] for j in stage_ids]
            
            print(f"Stage {s+1}/{stages}: {len(stage_ids)} samples, label distribution: {np.unique(stage_labels, return_counts=True)}, e.g., {stage_ids[0], stage_labels[0]}")
        
            # Use StratifiedKFold to ensure class balance
            skf_clients = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)        

            for i, (others_data_idx, train_data_idx) in enumerate(skf_clients.split(stage_ids, stage_labels)):
                client_ids = [stage_ids[j] for j in train_data_idx]
                client_labels = [labels[j] for j in client_ids]
                print(f"  Client {i+1}/{num_clients}: {len(client_ids)} samples, label distribution: {np.unique([labels[j] for j in client_ids], return_counts=True)}, e.g., {client_ids[0], client_labels[0]}")
                train_val_ids, test_ids = train_test_split(client_ids, test_size=test_p, random_state=seed+i, stratify=client_labels)
                
                skf_cross_fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed+i)
                
                for j, (train_idx, val_idx) in enumerate(skf_cross_fold.split(train_val_ids, [labels[id] for id in train_val_ids])):
                    train_ids = [train_val_ids[k] for k in train_idx]
                    val_ids = [train_val_ids[k] for k in val_idx]
                    
                    # Print statistics
                    if verbose:
                        print(f"Stage {s+1}/{stages}, Client {i+1}/{num_clients}, Fold {j+1}/{folds}:")
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
    else:
        raise NotImplementedError("Non-balanced split not implemented yet.")

def create_csv_split(clients: List[Tuple[List[int], List[int], List[int]]], num_clients:int, num_folds: int, num_stages: int, name:str, outputs_dir: str):
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
    
    len_splits = len(clients)
    assert len_splits == num_clients * num_folds * num_stages, f"Number of splits {len_splits} does not match num_clients * num_folds * num_stages"
    for stage in range(num_stages):
        for client in range(num_clients):
            for fold in range(num_folds):
                split_data = {name: [] for name in split_names}
                print(f"Creating CSV for client {client}, fold {fold}")
                split = clients[stage * num_clients * num_folds + client * num_folds + fold]
                for i, split_ids in enumerate(split):
                    split_data[split_names[i]].extend(split_ids)
                    if fold == 0 and i == 0: 
                        for id in split_data['train']:
                            assert id not in seen_ids, f"ID {id} is in the training set of multiple clients!"
                            seen_ids.add(id)
                        for id in split_data['val']:
                            assert id not in seen_ids, f"ID {id} is in the validation set of multiple clients!"
                            seen_ids.add(id)
                        for id in split_data['test']:
                            assert id not in seen_ids, f"ID {id} is in the test set of multiple clients!"
                            seen_ids.add(id)
        
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_data.items()]))
                df.to_csv(f"{outputs_dir}/splits_{client}_{fold}_{stage}.csv")
 
if __name__ == "__main__":
    num_clients = 3
    test_p = 0.1
    balance = True
    seed = 5
    folds = 5
    stages = 2
    output_dir = "splits/"

    splits_generator = create_k_clients_cross_fold_splits_stages(num_clients=num_clients, test_p=test_p, folds=folds, stages=stages, balance=balance, seed=seed, verbose=True, as_filename=True)
    
    splits = list(splits_generator)
    print(f"Generated {len(splits)} splits:")
    print(splits)
    create_csv_split(splits, name=f"chimera_{num_clients}_{folds}_{stages}_{test_p}_{seed}", num_clients=num_clients, num_folds=folds, num_stages=stages, outputs_dir=output_dir)
