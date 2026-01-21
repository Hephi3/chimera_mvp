#!/usr/bin/env python3
"""
Multi-Experiment Comparison Script

Compares F1 scores and confusion matrices between multiple experiments 
that use the same ensemble strategy across different splits.
"""

import os
import json
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_test_samples(csv_path: str, split_num: int = 0) -> Set[str]:
    """Load test sample names from CSV file for a specific split."""
    test_samples = set()
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            if 'test' not in reader.fieldnames:
                print("Warning: 'test' column not found in CSV file")
                return test_samples
            
            for row in reader:
                test_sample = row.get('test', '').strip()
                if test_sample:
                    sample_name = test_sample.replace('_HE', '')
                    test_samples.add(sample_name)
                    
    except Exception as e:
        print(f"Error loading test samples from {csv_path}: {e}")
        
    return test_samples

def is_test_interface(interface_dir: str, base_dir: str, test_samples: Set[str]) -> bool:
    """Check if an interface belongs to the test set."""
    if not test_samples:
        return True
        
    input_dir = os.path.join(base_dir, 'input')
    images_dir = os.path.join(input_dir, interface_dir, 'images', 'bladder-cancer-tissue-biopsy-wsi')
    
    if not os.path.exists(images_dir):
        return False
        
    try:
        tif_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    except OSError:
        return False
    
    if not tif_files:
        return False
        
    for tif_file in tif_files:
        tif_name = os.path.splitext(tif_file)[0]
        if tif_name in test_samples:
            return True
            
    return False

def load_results_from_directory(output_dir: str, test_samples: Optional[Set[str]] = None) -> List[Dict]:
    """Load all prediction results from a specific output directory."""
    results = []
    base_dir = os.path.dirname(output_dir)
    
    for interface_dir in sorted(os.listdir(output_dir)):
        if not interface_dir.startswith('interface_'):
            continue
            
        if test_samples is not None and not is_test_interface(interface_dir, base_dir, test_samples):
            continue
            
        prediction_file = os.path.join(output_dir, interface_dir, 'brs-probability.json')
        
        if os.path.exists(prediction_file):
            try:
                with open(prediction_file, 'r') as f:
                    data = json.load(f)
                    
                    probability = data.get('probability', None)
                    true_label = data.get('label', None)
                    
                    if probability is not None and true_label is not None:
                        if isinstance(probability, list):
                            probability = probability[-1]
                        
                        # Binary classification: BRS3 vs not-BRS3
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
    """Calculate F1 score and confusion matrix."""
    if not results:
        return {}
    
    y_true = [r['true_binary'] for r in results]
    y_pred = [r['pred_binary'] for r in results]
    
    # Confusion matrix components
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    total = len(results)
    
    # Calculate F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    
    return {
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }

def parse_directory_name(dir_name: str) -> Tuple[str, str]:
    """Parse directory name to extract experiment and split."""
    # Example: output_18-12-49_0_026_splits_split1_average
    if dir_name.startswith('output_'):
        parts = dir_name[7:]  # Remove 'output_' prefix
    else:
        parts = dir_name
    
    # Split by underscores and find split
    components = parts.split('_')
    
    split_idx = None
    for i, comp in enumerate(components):
        if comp.startswith('split') and comp != 'splits':
            split_idx = i
            break
    
    if split_idx is not None:
        experiment = '_'.join(components[:split_idx])
        split = components[split_idx]
    else:
        experiment = '_'.join(components[:-1])
        split = 'split0'
    
    return experiment, split

def load_experiment_results(base_test_dir: str, experiments: List[str], 
                          strategy: str, test_samples: Optional[Set[str]] = None) -> Dict[str, Dict]:
    """Load results for multiple specific experiments with the same strategy."""
    results = {}
    
    # Look for directories matching the pattern
    for item in os.listdir(base_test_dir):
        if not item.startswith('output_'):
            continue
            
        # Check if this directory matches one of our experiments
        if any(exp in item for exp in experiments) and strategy in item:
            output_dir = os.path.join(base_test_dir, item)
            
            if os.path.exists(output_dir):
                experiment, split = parse_directory_name(item)
                
                # Only process if it's one of our target experiments
                if experiment in experiments:
                    print(f"Loading results from: {experiment} - {split}")
                    
                    prediction_results = load_results_from_directory(output_dir, test_samples)
                    
                    if prediction_results:
                        metrics = calculate_metrics(prediction_results)
                        key = f"{experiment}_{split}"
                        results[key] = {
                            'experiment': experiment,
                            'split': split,
                            'results': prediction_results,
                            'metrics': metrics
                        }
                        print(f"  Loaded {len(prediction_results)} samples, F1: {metrics['f1_score']:.3f}")
                    else:
                        print(f"  No results found")
    
    return results

def create_comparison_plot(results: Dict[str, Dict], experiments: List[str], 
                         strategy: str, output_dir: str):
    """Create comparison plots for F1 scores and confusion matrices."""
    
    # Separate results by experiment
    exp_results = {}
    for exp in experiments:
        exp_results[exp] = {k: v for k, v in results.items() if v['experiment'] == exp}
    
    n_experiments = len(experiments)
    
    # Create figure with dynamic layout
    if n_experiments == 2:
        fig = plt.figure(figsize=(16, 10))
        cm_positions = [(2, 3, 4), (2, 3, 5)]
        summary_pos = (2, 3, (3, 6))
    elif n_experiments == 3:
        fig = plt.figure(figsize=(18, 12))
        cm_positions = [(3, 3, 4), (3, 3, 5), (3, 3, 6)]
        summary_pos = (3, 3, (7, 9))
    else:  # 4 or more
        fig = plt.figure(figsize=(20, 14))
        cm_positions = [(3, 4, 5), (3, 4, 6), (3, 4, 7), (3, 4, 8)]
        if n_experiments > 4:
            cm_positions.extend([(3, 4, i) for i in range(9, 9 + n_experiments - 4)])
        summary_pos = (3, 4, (9, 12))
    
    # F1 Score comparison
    ax1 = plt.subplot(2, max(3, n_experiments), (1, min(3, n_experiments)))
    
    splits = sorted(set([v['split'] for v in results.values()]))
    
    # Prepare data for all experiments
    exp_f1_data = {}
    for exp in experiments:
        exp_f1_scores = []
        for split in splits:
            exp_key = f"{exp}_{split}"
            exp_f1 = exp_results[exp].get(exp_key, {}).get('metrics', {}).get('f1_score', 0)
            exp_f1_scores.append(exp_f1)
        exp_f1_data[exp] = exp_f1_scores
    
    # Create bar plot
    x = np.arange(len(splits))
    width = 0.8 / n_experiments
    colors = plt.cm.Set1(np.linspace(0, 1, n_experiments))
    
    for i, (exp, f1_scores) in enumerate(exp_f1_data.items()):
        offset = (i - (n_experiments - 1) / 2) * width
        bars = ax1.bar(x + offset, f1_scores, width, 
                      label=f'{exp}', alpha=0.8, color=colors[i])
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Split')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'F1 Score Comparison\nStrategy: {strategy}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Confusion matrices for each experiment
    combined_results = {}
    for i, exp in enumerate(experiments):
        if i < len(cm_positions):
            ax_cm = plt.subplot(*cm_positions[i])
            combined = combine_confusion_matrices(exp_results[exp])
            combined_results[exp] = combined
            
            cmap = plt.cm.Blues if i == 0 else plt.cm.Oranges if i == 1 else plt.cm.Greens if i == 2 else plt.cm.Reds
            sns.heatmap(combined['confusion_matrix'], annot=True, fmt='d', cmap=cmap,
                        xticklabels=['Not-BRS3', 'BRS3'], yticklabels=['Not-BRS3', 'BRS3'], ax=ax_cm)
            
            # Truncate experiment name if too long
            exp_display = exp if len(exp) <= 20 else exp[:17] + "..."
            ax_cm.set_title(f'{exp_display}\nF1: {combined["f1_score"]:.3f}', fontsize=10)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
    
    # Summary statistics
    ax_summary = plt.subplot(*summary_pos)
    ax_summary.axis('off')
    
    # Calculate summary stats for all experiments
    summary_lines = [f"COMPARISON SUMMARY\n", f"Strategy: {strategy}", f"Number of splits: {len(splits)}\n"]
    
    for i, exp in enumerate(experiments):
        exp_f1_scores = exp_f1_data[exp]
        mean_f1 = np.mean(exp_f1_scores) if exp_f1_scores else 0
        std_f1 = np.std(exp_f1_scores) if len(exp_f1_scores) > 1 else 0
        combined_f1 = combined_results[exp]['f1_score']
        total_samples = combined_results[exp]['total']
        
        exp_display = exp if len(exp) <= 25 else exp[:22] + "..."
        summary_lines.extend([
            f"EXPERIMENT {i+1}: {exp_display}",
            f"• Mean F1: {mean_f1:.3f} ± {std_f1:.3f}",
            f"• Combined F1: {combined_f1:.3f}",
            f"• Total samples: {total_samples}\n"
        ])
    
    # Find best experiment
    best_exp = max(experiments, key=lambda exp: combined_results[exp]['f1_score'])
    best_f1 = combined_results[best_exp]['f1_score']
    
    summary_lines.extend([
        "PERFORMANCE RANKING:",
        "────────────────────"
    ])
    
    # Sort experiments by combined F1 score
    sorted_experiments = sorted(experiments, key=lambda exp: combined_results[exp]['f1_score'], reverse=True)
    for i, exp in enumerate(sorted_experiments):
        f1 = combined_results[exp]['f1_score']
        rank_symbol = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        exp_display = exp if len(exp) <= 25 else exp[:22] + "..."
        summary_lines.append(f"{rank_symbol} {exp_display}: {f1:.3f}")
    
    summary_text = "\n".join(summary_lines)
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=10, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'multi_experiment_comparison_{strategy}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def combine_confusion_matrices(experiment_results: Dict[str, Dict]) -> Dict:
    """Combine confusion matrices and metrics across all splits for an experiment."""
    if not experiment_results:
        return {'confusion_matrix': [[0, 0], [0, 0]], 'f1_score': 0, 'total': 0}
    
    # Combine all TP, TN, FP, FN across splits
    total_tp = sum(data['metrics']['tp'] for data in experiment_results.values())
    total_tn = sum(data['metrics']['tn'] for data in experiment_results.values())
    total_fp = sum(data['metrics']['fp'] for data in experiment_results.values())
    total_fn = sum(data['metrics']['fn'] for data in experiment_results.values())
    
    # Recalculate metrics on combined data
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    total = total_tp + total_tn + total_fp + total_fn
    
    return {
        'confusion_matrix': [[total_tn, total_fp], [total_fn, total_tp]],
        'f1_score': f1_score,
        'total': total,
        'precision': precision,
        'recall': recall
    }

def print_detailed_comparison(results: Dict[str, Dict], experiments: List[str], strategy: str):
    """Print detailed comparison results."""
    print("\n" + "="*80)
    print("MULTI-EXPERIMENT COMPARISON - DETAILED RESULTS")
    print("="*80)
    
    # Separate results by experiment
    exp_results = {}
    for exp in experiments:
        exp_results[exp] = {k: v for k, v in results.items() if v['experiment'] == exp}
    
    print(f"\nExperiments ({len(experiments)}):")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    print(f"Strategy: {strategy}")
    print(f"Total splits found: {len(set([v['split'] for v in results.values()]))}")
    
    print(f"\n" + "-"*80)
    print("SPLIT-BY-SPLIT COMPARISON")
    print("-"*80)
    
    # Create comparison table
    splits = sorted(set([v['split'] for v in results.values()]))
    
    # Dynamic column width based on number of experiments
    col_width = max(10, min(15, 80 // (len(experiments) + 2)))
    
    header = f"{'Split':<{col_width}}"
    for i, exp in enumerate(experiments):
        exp_short = exp[:col_width-3] + "..." if len(exp) > col_width else exp
        header += f"{exp_short:<{col_width}}"
    header += f"{'Best':<{col_width}}"
    
    print(header)
    print("-" * len(header))
    
    for split in splits:
        row = f"{split:<{col_width}}"
        split_f1_scores = {}
        
        for exp in experiments:
            exp_key = f"{exp}_{split}"
            f1_score = exp_results[exp].get(exp_key, {}).get('metrics', {}).get('f1_score', 0)
            split_f1_scores[exp] = f1_score
            row += f"{f1_score:<{col_width}.3f}"
        
        # Find best experiment for this split
        best_exp = max(split_f1_scores.keys(), key=lambda x: split_f1_scores[x]) if split_f1_scores else ""
        best_short = best_exp[:col_width-3] + "..." if len(best_exp) > col_width else best_exp
        row += f"{best_short:<{col_width}}"
        
        print(row)
    
    # Combined results
    print(f"\n" + "-"*80)
    print("COMBINED RESULTS (All Splits)")
    print("-"*80)
    
    combined_results = {}
    for exp in experiments:
        combined_results[exp] = combine_confusion_matrices(exp_results[exp])
    
    for i, exp in enumerate(experiments, 1):
        combined = combined_results[exp]
        print(f"\nExperiment {i} ({exp}):")
        print(f"  Total samples: {combined['total']}")
        print(f"  F1 Score: {combined['f1_score']:.3f}")
        print(f"  Precision: {combined['precision']:.3f}")
        print(f"  Recall: {combined['recall']:.3f}")
        print(f"  Confusion Matrix: {combined['confusion_matrix']}")
    
    # Calculate cross-split statistics
    print(f"\n" + "-"*80)
    print("CROSS-SPLIT STATISTICS")
    print("-"*80)
    
    print(f"{'Experiment':<30} {'Mean F1':<10} {'Std F1':<10} {'Min F1':<10} {'Max F1':<10}")
    print("-" * 80)
    
    stats_data = []
    for exp in experiments:
        exp_f1_scores = []
        for split in splits:
            exp_key = f"{exp}_{split}"
            f1_score = exp_results[exp].get(exp_key, {}).get('metrics', {}).get('f1_score', 0)
            if f1_score > 0:  # Only include valid scores
                exp_f1_scores.append(f1_score)
        
        if exp_f1_scores:
            mean_f1 = np.mean(exp_f1_scores)
            std_f1 = np.std(exp_f1_scores) if len(exp_f1_scores) > 1 else 0
            min_f1 = np.min(exp_f1_scores)
            max_f1 = np.max(exp_f1_scores)
            
            exp_display = exp[:28] + ".." if len(exp) > 30 else exp
            print(f"{exp_display:<30} {mean_f1:<10.3f} {std_f1:<10.3f} {min_f1:<10.3f} {max_f1:<10.3f}")
            
            stats_data.append({
                'experiment': exp,
                'mean_f1': mean_f1,
                'combined_f1': combined_results[exp]['f1_score'],
                'std_f1': std_f1
            })
    
    print(f"\n" + "-"*80)
    print("FINAL RANKING")
    print("-"*80)
    
    # Sort by combined F1 score
    stats_data.sort(key=lambda x: x['combined_f1'], reverse=True)
    
    print(f"{'Rank':<5} {'Experiment':<30} {'Combined F1':<12} {'Mean F1':<12} {'Consistency':<12}")
    print("-" * 80)
    
    for i, data in enumerate(stats_data, 1):
        exp = data['experiment']
        combined_f1 = data['combined_f1']
        mean_f1 = data['mean_f1']
        consistency = "High" if data['std_f1'] < 0.02 else "Medium" if data['std_f1'] < 0.05 else "Low"
        
        rank_symbol = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        exp_display = exp[:28] + ".." if len(exp) > 30 else exp
        
        print(f"{rank_symbol:<5} {exp_display:<30} {combined_f1:<12.3f} {mean_f1:<12.3f} {consistency:<12}")
    
    if stats_data:
        winner = stats_data[0]
        print(f"\n🏆 OVERALL WINNER: {winner['experiment']}")
        print(f"   Combined F1 Score: {winner['combined_f1']:.3f}")
        print(f"   Mean F1 Score: {winner['mean_f1']:.3f}")
        print(f"   Consistency: {'High' if winner['std_f1'] < 0.02 else 'Medium' if winner['std_f1'] < 0.05 else 'Low'}")
        
        if len(stats_data) > 1:
            runner_up = stats_data[1]
            improvement = winner['combined_f1'] - runner_up['combined_f1']
            print(f"   Better than runner-up by: {improvement:.3f} F1 points")

def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiments")
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                       help="List of experiment names (e.g., '18-12-49_0_026_splits' '18-12-52_1_002_splits')")
    parser.add_argument('--strategy', type=str, default='average',
                       help="Ensemble strategy to compare (default: 'average')")
    parser.add_argument('--test-csv', type=str,
                       default="/gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection/chimera_1_10_0.1/splits_0.csv",
                       help="Path to CSV file with train/val/test split")
    parser.add_argument('--output-dir', type=str, default='/gris/gris-f/homelv/phempel/masterthesis/test',
                       help="Output directory for plots")
    parser.add_argument('--test-only', action='store_true',
                       help="Analyze only test data (requires --test-csv)")
    
    args = parser.parse_args()
    
    base_test_dir = '/gris/gris-f/homelv/phempel/masterthesis/test'
    
    print("Multi-Experiment Comparison Analysis")
    print("=" * 50)
    print(f"Experiments ({len(args.experiments)}):")
    for i, exp in enumerate(args.experiments, 1):
        print(f"  {i}. {exp}")
    print(f"Strategy: {args.strategy}")
    
    if len(args.experiments) < 2:
        print("Error: At least 2 experiments are required for comparison.")
        return
    
    if len(args.experiments) > 6:
        print("Warning: Comparing more than 6 experiments may result in cluttered visualizations.")
    
    # Load test samples if specified
    test_samples = None
    if args.test_only and args.test_csv:
        print(f"\nTest-only mode enabled. Loading test samples from: {args.test_csv}")
        test_samples = load_test_samples(args.test_csv)
        print(f"Found {len(test_samples)} test samples")
    
    # Load results for all experiments
    print(f"\nLoading results...")
    results = load_experiment_results(base_test_dir, args.experiments, args.strategy, test_samples)
    
    if not results:
        print("No results found! Check experiment names and strategy.")
        print(f"Looking for directories containing:")
        for exp in args.experiments:
            print(f"  - '{exp}' and '{args.strategy}'")
        return
    
    print(f"\nLoaded results for {len(results)} experiment-split combinations")
    
    # Verify we have results for all experiments
    found_experiments = set([data['experiment'] for data in results.values()])
    missing_experiments = set(args.experiments) - found_experiments
    
    if missing_experiments:
        print(f"\nWarning: No results found for experiments: {list(missing_experiments)}")
        print("Proceeding with available experiments...")
        
    if len(found_experiments) < 2:
        print("Error: Need at least 2 experiments with results for comparison.")
        return
    
    # Print detailed comparison
    print_detailed_comparison(results, list(found_experiments), args.strategy)
    
    # Create comparison plot
    print(f"\nGenerating comparison plot...")
    create_comparison_plot(results, list(found_experiments), args.strategy, args.output_dir)
    
    plot_file = os.path.join(args.output_dir, f'multi_experiment_comparison_{args.strategy}.png')
    print(f"\nAnalysis completed!")
    print(f"Plot saved to: {plot_file}")

if __name__ == "__main__":
    main()
