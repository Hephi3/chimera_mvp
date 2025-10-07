from random import random
import numpy as np
import torch
from utils.clam_utils import get_split_loader, get_optim, print_network, calculate_error
import os
from dataset.clam_dataset.dataset_generic import save_splits
from torch import optim

from models.mm_models.multimodal_hierarchical import MultimodalHierarchical
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # print("Used this path")
        # roc_auc = roc_auc_score(binary_labels, [p[0][pos_class_idx] for p in all_probs])

        raise ValueError("Invalid format for all_probs. Expected a list of probabilities or logits.")
    f1 = f1_score(binary_labels, np.array(all_preds) == pos_class_idx)
    binary_acc = accuracy_score(binary_labels, np.array(all_preds) == pos_class_idx)
    
    # Log metrics to TensorBoard'
    if writer is not None:
        if loss is not None: writer.add_scalar(f'Loss/{kind}/{submodel}', loss, epoch)
        writer.add_scalar(f'Accuracy/{kind}/{submodel}', acc, epoch)
        writer.add_scalar(f'ROC_AUC/{kind}/{submodel}', roc_auc, epoch) # Binary
        writer.add_scalar(f'F1/{kind}/{submodel}', f1, epoch) # Binary
        writer.add_scalar(f'Binary_Accuracy/{kind}/{submodel}', binary_acc, epoch) # Binary
    
    return acc, roc_auc, f1

def train(datasets, cur, args, device):
    """   
        train for a single fold
    """
    
    seed = args.seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    verbose = not args.no_verbose
    if verbose: print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, "log")
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    writer_dir = os.path.join(writer_dir, f"fold_{cur}")
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    
    if verbose: print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    if verbose: print('Done!')
    if verbose: print("Training on {} samples".format(len(train_split)))
    if verbose: print("Validating on {} samples".format(len(val_split)))
    if verbose: print("Testing on {} samples".format(len(test_split)))

    if verbose: print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('Loss function {} not implemented'.format(args.bag_loss))
    if verbose: print('Done!')
    
    if verbose: print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None:
        model_dict.update({"size_arg": args.model_size})
    
    if args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    
    # if args.norm:
    #     model_dict.update({'norm': True})
    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    model = MultimodalHierarchical(**model_dict, instance_loss_fn=instance_loss_fn, num_levels=len(args.pages), clinical_dim=args.clinical_dim,
            norm=args.norm, 
            top_p=args.top_p)
    
    _ = model.to(device)
    if verbose: print('Done!')
    if verbose: print_network(model)

    if verbose: print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if verbose: print('Done!')
    
    if verbose: print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample, device=device)
    val_loader = get_split_loader(val_split, device=device)
    test_loader = get_split_loader(test_split, device=device)
    if verbose: print('Done!')

    if verbose: print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 5, stop_epoch=40, verbose = verbose)

    else:
        early_stopping = None
    if verbose: print('Done!')

    for epoch in tqdm(range(args.max_epochs), desc='Training Epochs'):#, disable=not verbose):
        if epoch < 15:
            phase = 'clam_only'
        elif epoch < 30:
            if args.clinical_dim != 23:
                phase = 'cd_only'
            else:
                phase = 'fusion'
        else:
            phase = 'fusion'
        model.clam.requires_grad_(phase in ['clam_only', 'fusion'])
        model.clinical_model.requires_grad_(phase in ['cd_only', 'fusion'])
        if hasattr(model, 'fusion_net'):
            model.fusion_net.requires_grad_(phase == 'fusion')
            model.classifier.requires_grad_(phase == 'fusion')
        train_loop_clam(epoch, model, train_loader, optimizer,args.bag_weight, writer, loss_fn, verbose=verbose, phase=phase, device=device)
        stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir, verbose=verbose, scheduler=scheduler, phase=phase, device=device)

        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # _, val_error, val_auc, _= summary(model, val_loader, args.n_classes, results_dir=args.results_dir, fold_nr=cur, type='val')
    # if verbose: print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    results_dict, test_error, test_auc, _ = summary(model, test_loader, args.n_classes, results_dir=args.results_dir, fold_nr=cur, device=device, type='test')
    if verbose: print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 
    return results_dict

def apply_model(loader_data, model, testing=False, plot_coords = False, device=None):
    data, label, coords, clinical_data, slide_id = loader_data
    clinical_data = clinical_data.to(device)
    if isinstance(data, list):
        data = [d.to(device) for d in data]
    else:
        data = data.to(device)
    label = label.to(device)
    coords = [c.to(device) for c in coords]
    slide_id = slide_id.to(device) if isinstance(slide_id, torch.Tensor) else slide_id 
    
    for i in range(3):
        assert len(data[i]) == len(coords[i]) or slide_id == "2A_001_HE", f"Mismatch in scale {i} lengths: collated[{i}]={len(data[i])},   coords[{i}]={len(coords[i])}, for slide {slide_id}"

    if testing:
        with torch.no_grad():
            results = model(data, coords=coords, clinical_features=clinical_data)
    else:
        results = model(data, label=label, coords=coords, clinical_features=clinical_data, instance_eval=True, slide_id=slide_id, plot_coords=plot_coords)

    return results, label

def train_loop_clam(epoch, model, loader, optimizer, bag_weight, writer = None, loss_fn = None, verbose = True, phase = None, device=None):
    # print information about loader data to compare with other experiment:
    # data, label, coords, clinical_data, slide_id = next(iter(loader))
    # print(f"Data batch shapes: {[d.shape for d in data]}, Labels: {label}, Coords: {[c.shape for c in coords]}, Clinical data: {clinical_data.shape}, Slide IDs: {slide_id}")
    
    # print(1/0)
    
    model.train()
    
    train_loss_mm = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    train_loss_clam = 0.
    all_probs_clam = []
    all_preds_clam = []
    
    train_loss_cd = 0.
    all_probs_cd = []
    all_preds_cd = []
    
    loss_cd = 0.

    if verbose: print('\n')
    
    for batch_idx, loader_data in enumerate(loader):
        
        result_dict, label = apply_model(loader_data, model, plot_coords = False, device=device)
        
        logits, Y_prob, Y_hat, _, _ = result_dict['MM']

        all_labels.append(label.item())
        
        # MM
        if isinstance(loss_fn, nn.NLLLoss):
            Y_log_prob = torch.log(Y_prob + 1e-8)
            loss_mm = loss_fn(Y_log_prob, label)
        else:
            loss_mm = loss_fn(logits, label)
        loss_value = loss_mm.item()
        train_loss_mm += loss_value
        Y_log_prob = torch.log(Y_prob + 1e-8)  # Add small value to avoid log(0)
        all_probs.append(Y_prob.cpu().detach().numpy())
        all_preds.append(Y_hat.item())
        
        # CLAM
        if isinstance(loss_fn, nn.NLLLoss):
            Y_log_prob = torch.log(result_dict['CLAM'][0] + 1e-8)
            loss_clam = loss_fn(Y_log_prob, label)
        else:
            loss_clam = loss_fn(result_dict['CLAM'][0], label)
        train_loss_clam += loss_clam.item()
        all_probs_clam.append(result_dict['CLAM'][1].cpu().detach().numpy())
        all_preds_clam.append(result_dict['CLAM'][2].item())
        instance_dict = result_dict['CLAM'][4]  # Get instance-related results from MM
        instance_loss_clam = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss_clam.item()
        train_inst_loss += instance_loss_value
        
        loss_clam = bag_weight * loss_clam + (1-bag_weight) * instance_loss_clam 
        
        # CD
        logits_cd = result_dict['CD'][0]
        if logits_cd is not None:
            if isinstance(loss_fn, nn.NLLLoss):
                Y_log_prob = torch.log(result_dict['CD'][0] + 1e-8)
                loss_cd = loss_fn(Y_log_prob, label)
            else:
                cd_results = result_dict['CD'][0].squeeze(0)
                loss_cd = loss_fn(result_dict['CD'][0], label)
            train_loss_cd += loss_cd.item()
            all_probs_cd.append(result_dict['CD'][1].cpu().detach().numpy())
            all_preds_cd.append(result_dict['CD'][2].item())
        
        w_mm = torch.exp(model.loss_weight_mm)
        w_clam = torch.exp(model.loss_weight_clam)
        w_cd = torch.exp(model.loss_weight_cd) if logits_cd is not None else 0.0

        
        # # redo:            
        if phase == 'clam_only':
            total_loss = loss_clam
        elif phase == 'cd_only':
            total_loss = loss_cd
        else:
            
            # Fusion + submodale Hilfsverluste
            norm = w_mm + w_clam + w_cd
            total_loss = (w_mm * loss_mm + w_clam * loss_clam + w_cd * loss_cd) / norm
        
        if verbose and (batch_idx + 1) % 20 == 0:
            print('batch {}, loss_mm: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}'.format(label.item()))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        

    # calculate loss and error for epoch
    train_loss_mm /= len(loader)
    train_error /= len(loader)
    train_loss_clam /= len(loader)
    train_loss_cd /= len(loader)
    
    log_metrics(writer, epoch, train_loss_mm, all_labels, all_preds, all_probs, 'train', 'MM')
    log_metrics(writer, epoch, train_loss_clam, all_labels, all_preds_clam, all_probs_clam, 'train', 'CLAM')
    log_metrics(writer, epoch, train_loss_cd, all_labels, all_preds_cd, all_probs_cd, 'train', 'CD')
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        if verbose: print('\n')

    if verbose: print('Epoch: {}, train_loss_mm: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss_mm, train_inst_loss,  train_error))
        

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None, verbose = True, scheduler=None, phase = None, device=None):
    model.eval()
    val_loss_mm = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    val_loss_clam = 0.
    all_probs_clam = []
    all_preds_clam = []
    
    val_loss_cd = 0.
    all_probs_cd = []
    all_preds_cd = []
    
    
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.inference_mode():
        for batch_idx, loader_data in enumerate(loader):
            result_dict, label = apply_model(loader_data, model, device=device)
        
            logits, Y_prob, Y_hat, _, _ = result_dict['MM']
            
            if isinstance(loss_fn, nn.NLLLoss):
                Y_log_prob = torch.log(Y_prob + 1e-8)
                loss = loss_fn(Y_log_prob, label)
            else:
                loss = loss_fn(logits, label)
            val_loss_mm += loss.item()
            all_labels.append(label.item())
            all_probs.append(Y_prob.cpu().detach().numpy())
            all_preds.append(Y_hat.item())
            
            if isinstance(loss_fn, nn.NLLLoss):
                Y_log_prob = torch.log(result_dict['CLAM'][0] + 1e-8)
                loss_clam = loss_fn(Y_log_prob, label)
            else:
                loss_clam = loss_fn(result_dict['CLAM'][0], label)
            val_loss_clam += loss_clam.item()
            all_probs_clam.append(result_dict['CLAM'][1].cpu().detach().numpy())
            all_preds_clam.append(result_dict['CLAM'][2].item())
            instance_dict = result_dict['CLAM'][4]
            
            logits_cd = result_dict['CD'][0]
            if logits_cd is not None:
                if isinstance(loss_fn, nn.NLLLoss):
                    Y_log_prob = torch.log(result_dict['CD'][0] + 1e-8)
                    loss_cd = loss_fn(Y_log_prob, label)
                else:
                    loss_cd = loss_fn(result_dict['CD'][0], label)
                val_loss_cd += loss_cd.item()
                all_probs_cd.append(result_dict['CD'][1].cpu().detach().numpy())
                all_preds_cd.append(result_dict['CD'][2].item())
            
            

            if instance_dict is not None:
                instance_loss = instance_dict['instance_loss']
                
                inst_count+=1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']

            prob[batch_idx] = Y_prob.cpu().detach().numpy()
            labels[batch_idx] = label.item()
            

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss_mm /= len(loader)
    val_loss_clam /= len(loader)
    val_loss_cd /= len(loader)
    
    if scheduler and phase == 'fusion':
        scheduler.step(val_loss_mm)
    
    log_metrics(writer, epoch, val_loss_mm, all_labels, all_preds, all_probs, 'val', 'MM')
    log_metrics(writer, epoch, val_loss_clam, all_labels, all_preds_clam, all_probs_clam, 'val', 'CLAM')
    log_metrics(writer, epoch, val_loss_cd, all_labels, all_preds_cd, all_probs_cd, 'val', 'CD')

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if verbose: print('\nVal Set, val_loss_mm: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss_mm, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_mm, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            if verbose: print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, results_dir=None, fold_nr=None, device=None, type='test'):
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    
    all_probs_clam = np.zeros((len(loader), n_classes))
    all_preds_clam = np.zeros(len(loader))
    
    all_probs_cd = np.zeros((len(loader), n_classes))
    all_preds_cd = np.zeros(len(loader))
    
    

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, loader_data in enumerate(loader):

        result_dict, label = apply_model(loader_data, model, device=device, testing=True)

        logits, Y_prob, Y_hat, A_raw, instance_dict = result_dict['MM']
        slide_id = slide_ids.iloc[batch_idx]

        probs = Y_prob.cpu().detach().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        all_probs_clam[batch_idx] = result_dict['CLAM'][1].cpu().detach().numpy()
        all_preds_clam[batch_idx] = result_dict['CLAM'][2].item()
        
        if result_dict['CD'][1] is not None:
            all_probs_cd[batch_idx] = result_dict['CD'][1].cpu().detach().numpy()
            all_preds_cd[batch_idx] = result_dict['CD'][2].item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    
    writer_dir = os.path.join(results_dir,"log", f"fold_{fold_nr}")
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)


    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)
    if type == 'test':
        log_metrics(writer, None, None, all_labels, all_preds, all_probs, 'test', 'MM')
        log_metrics(writer, None, None, all_labels, all_preds_clam, all_probs_clam, 'test', 'CLAM')
        log_metrics(writer, None, None, all_labels, all_preds_cd, all_probs_cd, 'test', 'CD')

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, None
