"""
Script to test model performance on synthetic samples generated from class prototypes.

This script:
1. Loads a trained model with weights from a checkpoint
2. Loads a dataset and splits it by class
3. Creates prototypes for each of the 3 classes (BRS1, BRS2, BRS3)
4. Generates n synthetic samples per class using the prototypes
5. Evaluates the model's performance on these synthetic samples
"""

import torch
import torch.nn as nn
import numpy as np
from utils.fl_utils import get_model, load_data
from utils.method_utils import Prototype
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test model on prototype-generated samples')
    
    # Model and data paths
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--split_dir', type=str, default='chimera_3_0.1',
                       help='Directory containing data splits')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number to use')
    
    # Model parameters
    parser.add_argument('--drop_out', type=float, default=0.5)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=1536)
    parser.add_argument('--model_size', type=str, default='tiny')
    parser.add_argument('--subtyping', action='store_true', default=True)
    parser.add_argument('--B', type=int, default=8, help='k_sample parameter')
    parser.add_argument('--inst_loss', type=str, default='ce', choices=['ce', 'svm'])
    parser.add_argument('--bag_loss', type=str, default='ce', choices=['ce', 'svm'])
    parser.add_argument('--norm', action='store_true', default=True)
    parser.add_argument('--top_p', type=float, default=0.3)
    parser.add_argument('--clinical_dim', type=int, default=256)
    
    # Dataset parameters
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--num_stages', type=int, default=1)
    parser.add_argument('--pages', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--return_coords', action='store_true', default=True)
    parser.add_argument('--augmentations', type=dict, default=None)
    parser.add_argument('--use_split_k', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_verbose', action='store_true', default=False)
    
    # Sampling parameters
    parser.add_argument('--num_samples_per_class', type=int, default=50,
                       help='Number of synthetic samples to generate per class')
    parser.add_argument('--variance_scale', type=float, default=1.0,
                       help='Scale factor for prototype variance when sampling')
    
    return parser.parse_args()


def collect_features_by_class(dataset, num_classes=3):
    """
    Collect all feature vectors from dataset organized by class.
    
    Returns:
        dict: {class_id: list of feature tensors}
    """
    features_by_class = {i: [] for i in range(num_classes)}
    
    print(f"Collecting features from dataset with {len(dataset)} samples...")
    
    for idx in range(len(dataset)):
        data, label, coords, clinical_data, slide_id = dataset[idx]
        
        # data is a list of feature tensors for different scales
        # We'll use the first scale for prototype generation
        if isinstance(data, list):
            features = data[0]  # Use first scale
        else:
            features = data
        
        # Convert to numpy if it's a tensor
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
        
        # Average pool the features if they're patch-level
        if len(features.shape) > 1:
            features = np.mean(features, axis=0)
        
        features_by_class[label].append(features)
    
    # Print statistics
    for class_id in range(num_classes):
        print(f"  Class {class_id}: {len(features_by_class[class_id])} samples")
    
    return features_by_class


def create_prototypes(features_by_class, num_classes=3):
    """
    Create prototypes for each class from feature collections.
    
    Returns:
        dict: {class_id: Prototype object}
    """
    prototypes = {}
    
    print("\nCreating prototypes for each class...")
    for class_id in range(num_classes):
        features = features_by_class[class_id]
        if len(features) == 0:
            print(f"  Warning: No features for class {class_id}")
            continue
        
        prototype = Prototype.from_data(features, label=class_id)
        prototypes[class_id] = prototype
        prototype.print_info()
    
    return prototypes


def generate_synthetic_samples(prototypes, num_samples_per_class, variance_scale=1.0):
    """
    Generate synthetic samples from prototypes.
    
    Returns:
        tuple: (features_array, labels_array) where features_array is shape (n_samples, feature_dim)
    """
    all_features = []
    all_labels = []
    
    print(f"\nGenerating {num_samples_per_class} synthetic samples per class...")
    
    for class_id, prototype in prototypes.items():
        # Generate samples
        samples = prototype.sample(num_samples=num_samples_per_class, 
                                   variance_scale=variance_scale)
        
        # Store features and labels
        all_features.append(samples)
        all_labels.extend([class_id] * num_samples_per_class)
        
        print(f"  Class {class_id}: Generated {len(samples)} samples with shape {samples.shape}")
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels


def evaluate_on_synthetic_samples(model, features, labels, clinical_dim, device, num_classes=3):
    """
    Evaluate model on synthetic samples.
    
    Args:
        model: The neural network model
        features: numpy array of shape (n_samples, feature_dim)
        labels: numpy array of shape (n_samples,)
        clinical_dim: dimension of clinical features
        device: torch device
        num_classes: number of classes
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    
    print("\nEvaluating model on synthetic samples...")
    
    with torch.no_grad():
        for i in range(len(features)):
            # Prepare features
            # Model expects list of features for different scales
            # We'll replicate the same features for all scales
            feature_tensor = torch.from_numpy(features[i]).float().unsqueeze(0)  # (1, feature_dim)
            
            # Replicate for 3 scales (matching the dataset structure)
            h_list = [feature_tensor.to(device) for _ in range(3)]
            
            # Create dummy clinical features
            clinical_features = torch.zeros(1, 23).to(device)  # (1, 23)
            
            # Create dummy coordinates (not used in inference)
            coords = [torch.zeros(1, 2).to(device) for _ in range(3)]
            
            # Forward pass
            results = model(h_list, coords=coords, clinical_features=clinical_features)
            
            # Extract predictions from MM (multimodal) results
            logits, Y_prob, Y_hat, _, _ = results['MM']
            
            all_preds.append(Y_hat.item())
            all_probs.append(Y_prob.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(labels, all_preds)
    f1_micro = f1_score(labels, all_preds, average='micro')
    f1_macro = f1_score(labels, all_preds, average='macro')
    f1_weighted = f1_score(labels, all_preds, average='weighted')
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, all_preds, average=None)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS ON SYNTHETIC SAMPLES")
    print("="*60)
    print(f"Total samples: {len(labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  Class {i}: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, all_preds, 
                                target_names=[f'Class {i}' for i in range(num_classes)]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, all_preds))
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': labels
    }


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    model = get_model(args, device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Filter out instance_loss_fn related keys
    filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                          if not k.startswith('clam.instance_loss_fn')}
    
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset (using first client's data as example)
    print(f"\nLoading dataset from split_dir: {args.split_dir}, fold: {args.fold}")
    train_datasets, val_datasets, test_datasets = load_data(0, args)
    
    # Use training dataset to create prototypes (you can also use test_dataset)
    dataset = train_datasets[0]  # First stage
    
    # Collect features by class
    features_by_class = collect_features_by_class(dataset, num_classes=args.n_classes)
    
    # Create prototypes
    prototypes = create_prototypes(features_by_class, num_classes=args.n_classes)
    
    # Generate synthetic samples
    synthetic_features, synthetic_labels = generate_synthetic_samples(
        prototypes, 
        num_samples_per_class=args.num_samples_per_class,
        variance_scale=args.variance_scale
    )
    
    # Evaluate model on synthetic samples
    results = evaluate_on_synthetic_samples(
        model, 
        synthetic_features, 
        synthetic_labels,
        clinical_dim=args.clinical_dim,
        device=device,
        num_classes=args.n_classes
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
