#!/usr/bin/env python3
"""
Split-based Experiment Comparison Script

Compares results from multiple experiments across different splits.
Analyzes confusion matrices, F1 scores, and other classification metrics.
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score
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

def load_results_from_directory(output_dir: str, test_samples: Optional[Set[str]] = None) -> Tuple[List[Dict], str]:
    """Load all prediction results from a specific output directory."""
    results = []
    base_dir = os.path.dirname(output_dir)
    
    # Extract experiment and strategy info from directory name
    dir_name = os.path.basename(output_dir)
    if dir_name.startswith('output_'):
        experiment_strategy = dir_name[7:]  # Remove 'output_' prefix
    else:
        experiment_strategy = dir_name
    
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
    
    return results, experiment_strategy

def calculate_detailed_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive classification metrics."""
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
    
    # Basic metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return {
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score,
        'brs3_count': sum(y_true),
        'not_brs3_count': total - sum(y_true),
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }

def parse_directory_name(dir_name: str) -> Tuple[str, str, str]:
    """Parse directory name to extract experiment, split, and strategy."""
    # Example: output_18-12-49_0_026_splits_split1_average
    if dir_name.startswith('output_'):
        parts = dir_name[7:]  # Remove 'output_' prefix
    else:
        parts = dir_name
    
    # Split by underscores and try to identify components
    components = parts.split('_')
    
    # Find split and strategy
    split_idx = None
    strategy = None
    
    for i, comp in enumerate(components):
        if comp.startswith('split') and comp != 'splits':
            split_idx = i
            break
    
    if split_idx is not None:
        # Everything before split is experiment
        experiment = '_'.join(components[:split_idx])
        split = components[split_idx]
        # Everything after split is strategy
        strategy = '_'.join(components[split_idx + 1:]) if split_idx + 1 < len(components) else ''
    else:
        # Fallback: assume last component is strategy
        experiment = '_'.join(components[:-1])
        split = 'split0'  # default
        strategy = components[-1] if components else ''
    
    return experiment, split, strategy

def load_all_split_experiments(base_test_dir: str, test_samples: Optional[Set[str]] = None) -> Dict[str, Dict]:
    """Load results from all split-based experiment directories."""
    all_results = {}
    
    # Define expected patterns - we'll discover them from the filesystem
    experiment_patterns = [
        '18-12-49_0_026_splits',
        # '18-12-52_1_002_splits', 
        # '18-12-49_0_023_splits',
        # 'gsMM_30-10-34-005_redo_splits',
        'advanced_hierarchical_tuning'
    ]
    
    # Scan for all directories matching the pattern
    for item in os.listdir(base_test_dir):
        if not item.startswith('output_'):
            continue
            
        # Check if this looks like a split-based experiment
        if 'splits_split' in item and any(exp in item for exp in experiment_patterns):
            output_dir = os.path.join(base_test_dir, item)
            
            if os.path.exists(output_dir):
                experiment, split, strategy = parse_directory_name(item)
                print(f"Loading results from: {experiment} - {split} - {strategy}")
                
                results, exp_strategy = load_results_from_directory(output_dir, test_samples)
                
                if results:
                    metrics = calculate_detailed_metrics(results)
                    key = f"{experiment}_{split}_{strategy}"
                    all_results[key] = {
                        'experiment': experiment,
                        'split': split,
                        'strategy': strategy,
                        'results': results,
                        'metrics': metrics,
                        'name': key
                    }
                    print(f"  Loaded {len(results)} samples")
                else:
                    print(f"  No results found")
            else:
                print(f"Directory not found: {output_dir}")
    
    return all_results

def create_comparison_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table of all metrics."""
    data = []
    
    for name, result_data in all_results.items():
        experiment = result_data['experiment']
        split = result_data['split']
        strategy = result_data['strategy']
        metrics = result_data['metrics']
        
        data.append({
            'Experiment': experiment,
            'Split': split,
            'Strategy': strategy,
            'Total_Samples': metrics['total'],
            'Accuracy': metrics['accuracy'],
            'Balanced_Accuracy': metrics['balanced_accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'F1_Score': metrics['f1_score'],
            'TP': metrics['tp'],
            'TN': metrics['tn'],
            'FP': metrics['fp'],
            'FN': metrics['fn']
        })
    
    return pd.DataFrame(data)

def plot_comparison_metrics(comparison_df: pd.DataFrame, output_dir: str):
    """Create comparison plots for different metrics across splits."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Split-based Experiment Comparison', fontsize=16, fontweight='bold')
    
    # F1 Score comparison by experiment and split
    ax1 = axes[0, 0]
    # Create a combined identifier for better visualization
    comparison_df['Exp_Split'] = comparison_df['Experiment'] + '_' + comparison_df['Split']
    pivot_f1 = comparison_df.pivot(index='Exp_Split', columns='Strategy', values='F1_Score')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'F1 Score'})
    ax1.set_title('F1 Score by Experiment-Split and Strategy')
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Experiment_Split')
    
    # Accuracy comparison
    ax2 = axes[0, 1]
    pivot_acc = comparison_df.pivot(index='Exp_Split', columns='Strategy', values='Accuracy')
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Accuracy'})
    ax2.set_title('Accuracy by Experiment-Split and Strategy')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Experiment_Split')
    
    # Split comparison for each experiment
    ax3 = axes[1, 0]
    experiments = comparison_df['Experiment'].unique()
    strategies = comparison_df['Strategy'].unique()
    
    # Create a bar plot showing F1 scores across splits for each experiment
    x_pos = np.arange(len(experiments))
    width = 0.15
    
    for i, strategy in enumerate(strategies):
        f1_scores = []
        for exp in experiments:
            exp_data = comparison_df[(comparison_df['Experiment'] == exp) & (comparison_df['Strategy'] == strategy)]
            avg_f1 = exp_data['F1_Score'].mean() if not exp_data.empty else 0
            f1_scores.append(avg_f1)
        
        ax3.bar(x_pos + i * width, f1_scores, width, label=strategy, alpha=0.8)
    
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Average F1 Score')
    ax3.set_title('Average F1 Score by Experiment and Strategy')
    ax3.set_xticks(x_pos + width * (len(strategies) - 1) / 2)
    ax3.set_xticklabels([exp.split('_')[1] for exp in experiments], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Split variance analysis
    ax4 = axes[1, 1]
    split_variance_data = []
    
    for exp in experiments:
        for strategy in strategies:
            exp_strategy_data = comparison_df[(comparison_df['Experiment'] == exp) & (comparison_df['Strategy'] == strategy)]
            if len(exp_strategy_data) > 1:
                f1_std = exp_strategy_data['F1_Score'].std()
                f1_mean = exp_strategy_data['F1_Score'].mean()
                split_variance_data.append({
                    'Experiment': exp.split('_')[1],
                    'Strategy': strategy,
                    'F1_Mean': f1_mean,
                    'F1_Std': f1_std
                })
    
    if split_variance_data:
        variance_df = pd.DataFrame(split_variance_data)
        for strategy in strategies:
            strategy_data = variance_df[variance_df['Strategy'] == strategy]
            ax4.scatter(strategy_data['F1_Mean'], strategy_data['F1_Std'], 
                       label=strategy, s=100, alpha=0.7)
            
            # Add experiment labels
            for _, row in strategy_data.iterrows():
                ax4.annotate(row['Experiment'], 
                            (row['F1_Mean'], row['F1_Std']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Mean F1 Score')
    ax4.set_ylabel('F1 Score Standard Deviation')
    ax4.set_title('F1 Score Consistency Across Splits')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'split_comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(all_results: Dict[str, Dict], output_dir: str):
    """Plot confusion matrices for all experiments, splits, and strategies."""
    
    # Group by experiment and strategy, showing different splits
    experiments = list(set([data['experiment'] for data in all_results.values()]))
    strategies = list(set([data['strategy'] for data in all_results.values()]))
    
    # Create a larger figure to accommodate potentially many subplots
    n_experiments = len(experiments)
    n_strategies = len(strategies)
    
    fig, axes = plt.subplots(n_experiments, n_strategies, figsize=(4 * n_strategies, 4 * n_experiments))
    fig.suptitle('Confusion Matrices by Experiment and Strategy (Multiple Splits)', fontsize=16, fontweight='bold')
    
    if n_experiments == 1 and n_strategies == 1:
        axes = np.array([[axes]])
    elif n_experiments == 1:
        axes = axes.reshape(1, -1)
    elif n_strategies == 1:
        axes = axes.reshape(-1, 1)
    
    for i, experiment in enumerate(sorted(experiments)):
        for j, strategy in enumerate(sorted(strategies)):
            # Find all results for this experiment-strategy combination
            matching_results = []
            for key, data in all_results.items():
                if data['experiment'] == experiment and data['strategy'] == strategy:
                    matching_results.append(data)
            
            if matching_results:
                # If multiple splits, combine them or show the first one
                if len(matching_results) == 1:
                    result_data = matching_results[0]
                    cm = result_data['metrics']['confusion_matrix']
                    f1_score = result_data['metrics']['f1_score']
                    split_info = result_data['split']
                else:
                    # Combine results from multiple splits
                    all_tp = sum(r['metrics']['tp'] for r in matching_results)
                    all_tn = sum(r['metrics']['tn'] for r in matching_results)
                    all_fp = sum(r['metrics']['fp'] for r in matching_results)
                    all_fn = sum(r['metrics']['fn'] for r in matching_results)
                    
                    cm = [[all_tn, all_fp], [all_fn, all_tp]]
                    
                    # Recalculate F1 score for combined data
                    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
                    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    split_info = f"{len(matching_results)} splits"
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Not-BRS3', 'BRS3'],
                           yticklabels=['Not-BRS3', 'BRS3'],
                           ax=axes[i, j])
                
                # Add F1 score and split info to the title
                axes[i, j].set_title(f'{experiment.split("_")[1]}\n{strategy}\n{split_info}\nF1: {f1_score:.3f}', fontsize=10)
                axes[i, j].set_xlabel('Predicted')
                axes[i, j].set_ylabel('True')
            else:
                axes[i, j].text(0.5, 0.5, 'No Data', 
                               transform=axes[i, j].transAxes, 
                               ha='center', va='center')
                axes[i, j].set_title(f'{experiment}\n{strategy}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_comparison(comparison_df: pd.DataFrame, all_results: Dict[str, Dict]):
    """Print detailed comparison results."""
    print("\n" + "="*80)
    print("SPLIT-BASED EXPERIMENT COMPARISON - DETAILED RESULTS")
    print("="*80)
    
    # Overall summary
    print(f"\nOverall Summary:")
    print(f"Number of experiments: {comparison_df['Experiment'].nunique()}")
    print(f"Number of splits: {comparison_df['Split'].nunique()}")
    print(f"Number of strategies: {comparison_df['Strategy'].nunique()}")
    print(f"Total combinations: {len(comparison_df)}")
    
    # Best performing combinations
    print(f"\n" + "-"*60)
    print("TOP PERFORMING COMBINATIONS")
    print("-"*60)
    
    # Sort by F1 score
    top_f1 = comparison_df.nlargest(5, 'F1_Score')
    print("\nTop 5 by F1 Score:")
    for _, row in top_f1.iterrows():
        print(f"  {row['Experiment']} + {row['Split']} + {row['Strategy']}: F1={row['F1_Score']:.3f}, "
              f"Acc={row['Accuracy']:.3f}, Bal_Acc={row['Balanced_Accuracy']:.3f}")
    
    # Cross-split analysis
    print(f"\n" + "-"*60)
    print("CROSS-SPLIT ANALYSIS")
    print("-"*60)
    
    for experiment in comparison_df['Experiment'].unique():
        print(f"\n{experiment.upper()}:")
        exp_data = comparison_df[comparison_df['Experiment'] == experiment]
        
        for strategy in exp_data['Strategy'].unique():
            strategy_data = exp_data[exp_data['Strategy'] == strategy]
            if len(strategy_data) > 1:
                print(f"  {strategy}: F1={strategy_data['F1_Score'].mean():.3f}±{strategy_data['F1_Score'].std():.3f} "
                      f"(across {len(strategy_data)} splits)")
            else:
                print(f"  {strategy}: F1={strategy_data['F1_Score'].iloc[0]:.3f} (single split)")
    
    # Strategy comparison across all splits
    print(f"\n" + "-"*60)
    print("STRATEGY COMPARISON (Average across all experiments and splits)")
    print("-"*60)
    
    for strategy in comparison_df['Strategy'].unique():
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  F1 Score:        {strategy_data['F1_Score'].mean():.3f} ± {strategy_data['F1_Score'].std():.3f}")
        print(f"  Accuracy:        {strategy_data['Accuracy'].mean():.3f} ± {strategy_data['Accuracy'].std():.3f}")
        print(f"  Balanced Acc:    {strategy_data['Balanced_Accuracy'].mean():.3f} ± {strategy_data['Balanced_Accuracy'].std():.3f}")
        print(f"  Precision:       {strategy_data['Precision'].mean():.3f} ± {strategy_data['Precision'].std():.3f}")
        print(f"  Recall:          {strategy_data['Recall'].mean():.3f} ± {strategy_data['Recall'].std():.3f}")
    
    # Detailed table
    print(f"\n" + "-"*60)
    print("DETAILED METRICS TABLE")
    print("-"*60)
    
    # Format the dataframe for better display
    display_df = comparison_df.copy()
    numeric_cols = ['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score']
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    
    print(display_df.to_string(index=False))

def save_results_to_csv(comparison_df: pd.DataFrame, output_path: str):
    """Save comparison results to CSV file."""
    comparison_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare split-based experiment results")
    parser.add_argument('--test-csv', type=str, 
                       default="/gris/gris-f/homelv/phempel/masterthesis/MMFL/splits/split_collection/chimera_1_10_0.1/splits_0.csv",
                       help="Path to CSV file with train/val/test split")
    parser.add_argument('--output-dir', type=str, default='/gris/gris-f/homelv/phempel/masterthesis/test',
                       help="Output directory for plots and results")
    parser.add_argument('--test-only', action='store_true',
                       help="Analyze only test data (requires --test-csv). If not set, analyzes all data.")
    
    args = parser.parse_args()
    
    base_test_dir = '/gris/gris-f/homelv/phempel/masterthesis/test'
    
    print("Split-based Experiment Comparison Analysis")
    print("=" * 50)
    
    # Load test samples if specified and test-only mode is enabled
    test_samples = None
    if args.test_only:
        if args.test_csv:
            print(f"Test-only mode enabled. Loading test samples from: {args.test_csv}")
            test_samples = load_test_samples(args.test_csv)
            print(f"Found {len(test_samples)} test samples")
            if test_samples:
                print(f"Sample test names: {sorted(list(test_samples))[:5]}{'...' if len(test_samples) > 5 else ''}")
        else:
            print("Error: --test-only mode requires --test-csv to be specified.")
            return
    else:
        print("Analyzing all data (not restricted to test set).")
        if args.test_csv:
            print(f"Note: Test CSV provided ({args.test_csv}) but --test-only not set, so ignoring test split.")
    
    # Load all split experiment results
    print("\nLoading results from all split-based experiments...")
    all_results = load_all_split_experiments(base_test_dir, test_samples)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"\nLoaded results from {len(all_results)} experiment-split-strategy combinations")
    
    # Create comparison table
    comparison_df = create_comparison_table(all_results)
    
    # Print detailed comparison
    print_detailed_comparison(comparison_df, all_results)
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison_metrics(comparison_df, args.output_dir)
    plot_confusion_matrices(all_results, args.output_dir)
    
    # Save results
    csv_output = os.path.join(args.output_dir, 'split_comparison_results.csv')
    save_results_to_csv(comparison_df, csv_output)
    
    print(f"\nAnalysis completed!")
    print(f"Data scope: {'Test data only' if args.test_only else 'All data'}")
    print(f"Plots saved to: {args.output_dir}")
    print(f"  - split_comparison_metrics.png")
    print(f"  - split_confusion_matrices.png")
    print(f"Results table saved to: {csv_output}")

if __name__ == "__main__":
    main()
