# from random import random
import numpy as np
import torch
from utils.clam_utils import get_split_loader, get_optim, print_network, calculate_error
import os
from dataset.clam_dataset.dataset_generic import save_splits
from torch import optim

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import random



def log_metrics(writer, epoch, loss, all_labels, all_preds, all_probs, kind:str, submodel:str):
    if loss == 0:
        return
    labels = np.unique(all_labels)
    pos_class_idx = 2 if len(labels) == 3 else 1
    
    acc = accuracy_score(all_labels, all_preds)
    # Convert to one-vs-rest for AUC calculation (0 and 1 vs 2)
    binary_labels = np.array(all_labels) == pos_class_idx
    if type(all_probs[0]) == np.ndarray and type(all_probs[0][0]) == np.ndarray and len(all_probs[0][0]) == pos_class_idx + 1:
        roc_auc = roc_auc_score(binary_labels, [p[0][pos_class_idx] for p in all_probs] if type(all_probs[0]) == np.ndarray else all_probs)
    elif type(all_probs[0]) == np.ndarray and len(all_probs[0]) == pos_class_idx + 1:
        roc_auc = roc_auc_score(binary_labels, [p[pos_class_idx] for p in all_probs])
    elif type(all_probs) == list and type(all_probs[0]) == float:   
        roc_auc = roc_auc_score(binary_labels, all_probs)
    else:
        raise ValueError("Invalid format for all_probs. Expected a list of probabilities or logits.")
    f1 = f1_score(binary_labels, np.array(all_preds) == pos_class_idx)
    binary_acc = accuracy_score(binary_labels, np.array(all_preds) == pos_class_idx)
    
    # Log metrics to TensorBoard
    if writer is not None:
        if loss is not None: writer.add_scalar(f'Loss/{kind}/{submodel}', loss, epoch)
        writer.add_scalar(f'Accuracy/{kind}/{submodel}', acc, epoch)
        writer.add_scalar(f'ROC_AUC/{kind}/{submodel}', roc_auc, epoch) # Binary
        writer.add_scalar(f'F1/{kind}/{submodel}', f1, epoch) # Binary
        writer.add_scalar(f'Binary_Accuracy/{kind}/{submodel}', binary_acc, epoch) # Binary
    
    return acc, roc_auc, f1

def get_writer_dir(client_nr, round_nr, results_dir):
    writer_dir = os.path.join(results_dir, "log")
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    if type(client_nr) == int:
        writer_dir = os.path.join(writer_dir, f"client_{client_nr}_round_{round_nr}")
    else:
        writer_dir = os.path.join(writer_dir, f"server_round_{round_nr}")
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    return writer_dir

def train(model, train_split, val_split, args, cur, device, round_num=None, use_phases = True):
    return 0, 0

def apply_model(loader_data):
    data, label, coords, clinical_data, slide_id = loader_data
    return label

def test(model, test_split, args, device, results_dir=None, client_nr=None, n_classes=3, round_nr=None):
    test_loader = get_split_loader(test_split, device=device, args=args)
    loss, f1 = test_clam(model, test_loader, device=device, args=args, results_dir=results_dir, client_nr=client_nr, n_classes=n_classes,  round_nr=round_nr)
    return loss, f1

def test_clam(model, loader, device, args, results_dir=None, client_nr=None, n_classes=3, round_nr=None):
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    writer_dir = get_writer_dir(client_nr, round_nr, results_dir)
    
    for batch_idx, loader_data in enumerate(loader):
        
        label = apply_model(loader_data)
        # Predict 1 with probability prob_1
        prob_1 = 0.5
        random.seed(args.seed * 12345 + batch_idx + round_nr*100 + (client_nr if type(client_nr) == int else 42) + writer_dir.__hash__())
        random_pred = random.random()
        # print("RANDOM PRED:", random_pred, "From client_nr:", client_nr, "round_nr:", round_nr, "batch_idx:", batch_idx)
        prediction = random_pred < prob_1
        # Simulate logits, Y_prob, Y_hat, A_raw, and results_dict
        # Logits: Basically one hot vector with a single 1 at the index of the predicted class
        
        Y_hat = 2 if prediction else 0
        Y_prob = [1-Y_hat,0, Y_hat]

        probs = Y_prob
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label
        all_preds[batch_idx] = Y_hat

    
    
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)

    acc, roc_auc, f1 = log_metrics(writer, None, None, all_labels, all_preds, all_probs, 'test', 'MM')
    writer.flush()  # Ensure MM metrics are written
    # Ensure TensorBoard writer flushes all data before returning
    if writer is not None:
        writer.flush()
        writer.close()  # Properly close the writer to ensure data persistence
    return 0, f1