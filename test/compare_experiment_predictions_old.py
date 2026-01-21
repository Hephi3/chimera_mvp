#!/usr/bin/env python3
"""
Simple BRS Classification Results Analysis Script

Analyzes BRS probability predictions and calculates accuracy and F1 scores.
- The 'label' in brs-probability.json is the TRUE label
- Predicted label: BRS3 if probability > 0.5, else not-BRS3
- Binary classification: BRS3 vs not-BRS3

Usage: python analyze_brs_simple.py [--test-csv PATH]
"""

import os
import json
import argparse
import csv
from typing import Dict, List, Set, Optional

from matplotlib.pylab import rint

def load_test_samples(csv_path: str) -> Set[str]:
    """Load test sample names from CSV file."""
    test_samples = set()
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Check if 'test' column exists
            if 'test' not in reader.fieldnames:
                print("Warning: 'test' column not found in CSV file")
                return test_samples
            
            for row in reader:
                test_sample = row.get('test', '').strip()
                if test_sample:  # Skip empty values
                    # Remove _HE suffix if present to match with tif filenames
                    sample_name = test_sample.replace('_HE', '')
                    test_samples.add(sample_name)
                    
    except Exception as e:
        print(f"Error loading test samples from {csv_path}: {e}")
        
    return test_samples

def is_test_interface(interface_dir: str, base_dir: str, test_samples: Set[str]) -> bool:
    """Check if an interface belongs to the test set by looking at the tif filename."""
    if not test_samples:  # If no test samples loaded, include all
        return True
        
    # Look for tif file in the input directory structure
    input_dir = os.path.join(base_dir, 'input')
    images_dir = os.path.join(input_dir, interface_dir, 'images', 'bladder-cancer-tissue-biopsy-wsi')
    
    if not os.path.exists(images_dir):
        return False
        
    # Find tif files in the directory
    try:
        tif_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    except OSError:
        return False
    
    if not tif_files:
        return False
        
    # Check if any tif filename (without extension) matches test samples
    for tif_file in tif_files:
        tif_name = os.path.splitext(tif_file)[0]  # Remove .tif extension
        if tif_name in test_samples:
            return True
            
    return False

def load_all_results(output_dir: str, test_samples: Optional[Set[str]] = None) -> List[Dict]:
    """Load all prediction results from brs-probability.json files."""
    results = []
    if test_samples:
        print("Load only test samples!")
    
    # Determine base directory (parent of output_dir)
    base_dir = os.path.dirname(output_dir)
    
    for interface_dir in sorted(os.listdir(output_dir)):
        if not interface_dir.startswith('interface_'):
            continue
            
        # Filter by test set if specified
        if test_samples is not None and not is_test_interface(interface_dir, base_dir, test_samples):
            continue
            
        prediction_file = os.path.join(output_dir, interface_dir, 'brs-probability.json')
        
        if os.path.exists(prediction_file):
            try:
                with open(prediction_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract relevant data
                    probability = data.get('probability', None)
                    true_label = data.get('label', None)
                    
                    if probability is not None and true_label is not None:
                        if isinstance(probability, list):
                            probability = probability[-1]  
                        # Binary classification
                        true_binary = 1 if true_label == 'BRS3' else 0
                        pred_binary = 1 if probability > 0.5 else 0
                        
                        results.append({
                            'interface_id': interface_dir,
                            'probability': probability,
                            'true_label': true_label,
                            'true_binary': true_binary,
                            'pred_binary': pred_binary,
                            'correct': true_binary == pred_binary
                        })
                        
            except Exception as e:
                print(f"Warning: Could not load {prediction_file}: {e}")
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate classification metrics."""
    if not results:
        return {}
    
    # Extract arrays
    y_true = [r['true_binary'] for r in results]
    y_pred = [r['pred_binary'] for r in results]
    
    # Calculate confusion matrix components
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)  # True Positives
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)  # True Negatives
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)  # False Positives
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)  # False Negatives
    
    total = len(results)
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'brs3_count': sum(y_true),
        'not_brs3_count': total - sum(y_true)
    }

def print_results(results: List[Dict], metrics: Dict):
    """Print analysis results."""
    print("\n" + "="*60)
    print("BRS CLASSIFICATION ANALYSIS")
    print("="*60)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {metrics['total']}")
    print(f"  BRS3 samples: {metrics['brs3_count']} ({metrics['brs3_count']/metrics['total']*100:.1f}%)")
    print(f"  Not-BRS3 samples: {metrics['not_brs3_count']} ({metrics['not_brs3_count']/metrics['total']*100:.1f}%)")
    
    # Detailed class distribution
    class_counts = {}
    for r in results:
        label = r['true_label']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\nDetailed Class Distribution:")
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        print(f"  {label}: {count} ({count/metrics['total']*100:.1f}%)")
    
    print(f"\nBinary Classification Metrics (BRS3 vs Not-BRS3):")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Not-BRS3  BRS3")
    print(f"True Not-BRS3    {metrics['tn']:4d}    {metrics['fp']:4d}")
    print(f"     BRS3        {metrics['fn']:4d}    {metrics['tp']:4d}")
    
    print(f"\nPrediction Summary:")
    correct_predictions = metrics['tp'] + metrics['tn']
    incorrect_predictions = metrics['fp'] + metrics['fn']
    print(f"  Correct predictions:   {correct_predictions} ({correct_predictions/metrics['total']*100:.1f}%)")
    print(f"  Incorrect predictions: {incorrect_predictions} ({incorrect_predictions/metrics['total']*100:.1f}%)")
    
    if metrics['fp'] > 0:
        print(f"  False Positives: {metrics['fp']} (predicted BRS3, actually not)")
    if metrics['fn'] > 0:
        print(f"  False Negatives: {metrics['fn']} (predicted not-BRS3, actually BRS3)")

def show_sample_predictions(results: List[Dict], n: int = 10):
    """Show sample predictions."""
    print(f"\nSample Predictions (first {min(n, len(results))}):")
    print("-" * 65)
    print(f"{'Interface':<12} {'Prob':<6} {'True':<5} {'Pred':<5} {'Correct'}")
    print("-" * 65)
    
    for i, r in enumerate(results[:n]):
        pred_label = "BRS3" if r['pred_binary'] == 1 else "other"
        correct_symbol = "✓" if r['correct'] else "✗"
        print(f"{r['interface_id']:<12} {r['probability']:<6.3f} {r['true_label']:<5} "
              f"{pred_label:<5} {correct_symbol}")


def show_prediction_diffs(results: List[Dict], model_names: List[str]):
    """Show prediction differences between experiments."""
    print(f"\nPrediction Differences:")

    all_interface_ids = set()
    different_or_wrong_interface_ids = set()
    model_1_preds = {}
    interface_preds = {}

    print("-" * 65)
    for i, exp in enumerate(results):
        print(f"Experiment {i + 1}: {exp['experiment']}")
        for r in exp['results']:
            all_interface_ids.add(r['interface_id'])
            if not r['correct']:
                different_or_wrong_interface_ids.add(r['interface_id'])
            if i == 0:
                model_1_preds[r['interface_id']] = r['correct']
                interface_preds[r['interface_id']] = [r['probability']]
            else:
                interface_preds[r['interface_id']].append(r['probability'])
                if r['correct'] != model_1_preds[r['interface_id']]:
                    different_or_wrong_interface_ids.add(r['interface_id'])
            # pred_label = "BRS3" if r['pred_binary'] == 1 else "other"
            # correct_symbol = "✓" if r['correct'] else "✗"
            # print(f"  Interface: {r['interface_id']}, True: {r['true_label']}, Pred: {pred_label} -> {correct_symbol}")

    print("-" * 65)
    header_line = f"{'Interface':<12} {'True':<5} " + " ".join([f"{name:<5}" for name in model_names])
    print(header_line)
    print("-" * 65)
    for id in sorted(list(all_interface_ids)):
        idx_of_interface_id = results[0]['results'].index(next(r for r in results[0]['results'] if r['interface_id'] == id))
        true_label = results[0]['results'][idx_of_interface_id]['true_label']
        row = []
        for i in range(len(model_names)):
            # symbol = "✓" if interface_preds[id][i] else "✗"
            row.append(f"{interface_preds[id][i]:<5}")
        print(f"{id:<12} {true_label:<5} " + " ".join(row))

    print("-" * 65)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Analyze BRS classification results")
    parser.add_argument('--test-csv', type=str, help="Path to CSV file with train/val/test split to filter only test samples")#, default = "/gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection/chimera_1_10_0.1/splits_0.csv")
    
    # parser.add_argument('--experiment_results', type=str, default='output')
    parser.add_argument('--output-dir', type=str, default='/gris/gris-f/homelv/phempel/masterthesis/test/',help="Directory containing prediction results")
    # parser.add_argument('--test_only', action='store_true', help="Flag to indicate if only test results should be analyzed")
    args = parser.parse_args()
    
    experiment_results = [
        'output_advanced_hierarchical_tuning_12-16-32_0_017_splits_split1_majority_vote',
        'output_advanced_hierarchical_tuning_12-16-32_0_021_splits_split1_majority_vote',
        'output_grid_search_hierarchical_vacation_18-12-49_0_026_splits_split1_majority_vote'#,
        # 'output_grid_search_MM_30-10-34_0_005_splits_split1_majority_vote'
        
        
    ]
    
    print("BRS Classification Analysis")
    print("=" * 40)
    
    # Load test samples if CSV path is provided
    test_samples = None
    if args.test_csv:
        print(f"Loading test samples from: {args.test_csv}")
        test_samples = load_test_samples(args.test_csv)
        print(f"Found {len(test_samples)} test samples")
        if test_samples:
            print(f"Sample test names: {sorted(list(test_samples))[:5]}{'...' if len(test_samples) > 5 else ''}")
    
    # Load data
    print("\nLoading prediction results...")
    if test_samples:
        print("Filtering to include only test set interfaces...")
    
    exp_results = []
    
    # Configuration
    for exp in experiment_results:
        output_dir = os.path.join(args.output_dir, exp)
        assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist!"
        
        results = load_all_results(output_dir, test_samples)
    
        dataset_type = "test set only" if test_samples else "all available"
        print(f"Loaded {len(results)} samples ({dataset_type})")
    
        if not results:
            print("No valid results found!")
            if test_samples:
                print("Note: No matching test samples found. Check if:")
                print("  1. The CSV file path is correct")
                print("  2. The test sample names match the .tif filenames (without _HE suffix)")
                print("  3. The input directory structure is correct")
            return
    
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(results)
        
        # Print results
        print_results(results, metrics)
        # Show samples
        # show_sample_predictions(results, 300)
        
        exp_results.append({
            'experiment': exp,
            'results': results,
            'metrics': metrics
        })

    show_prediction_diffs(exp_results, experiment_results)

    
    print(f"\nAnalysis completed! ({dataset_type})")

if __name__ == "__main__":
    main()
