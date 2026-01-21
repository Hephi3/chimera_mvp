#!/usr/bin/env python3
"""
Comprehensive BRS Analysis Across All Splits
This script analyzes BRS performance across all 5 splits using the existing compare_experiment_predictions.py
"""

import os
import sys
import subprocess
import json
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import warnings
warnings.filterwarnings('ignore')

def run_single_experiment_analysis(base_dir, experiment_name, split_num, test_csv=None):
    """Run the existing script for a single experiment and split"""
    
    # Construct the full experiment name
    full_experiment_name = f"{experiment_name}_split{split_num}_majority_vote"
    output_dir = os.path.join(base_dir, full_experiment_name)
    
    if not os.path.exists(output_dir):
        print(f"Warning: Directory {output_dir} does not exist, skipping...")
        return None
    
    # Import and use the existing script's functions
    test_dir = '/gris/gris-f/homelv/phempel/masterthesis/test'
    if test_dir not in sys.path:
        sys.path.append(test_dir)
    from compare_experiment_predictions import load_all_results, calculate_metrics, load_test_samples
    
    # Load test samples if provided
    test_samples = None
    if test_csv:
        test_samples = load_test_samples(test_csv)
    
    # Load results
    results = load_all_results(output_dir, test_samples)
    
    if not results:
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    return {
        'experiment': experiment_name,
        'split': split_num,
        'full_name': full_experiment_name,
        'results': results,
        'metrics': metrics
    }

def aggregate_cross_split_results(all_results):
    """Aggregate results across all splits for each experiment"""
    
    # Group by experiment base name
    experiments = {}
    for result in all_results:
        exp_name = result['experiment']
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(result)
    
    aggregated_results = {}
    
    for exp_name, exp_splits in experiments.items():
        print(f"\nProcessing {exp_name} across {len(exp_splits)} splits...")
        
        # Collect metrics from all splits
        split_metrics = []
        all_predictions = []
        
        for split_result in exp_splits:
            split_metrics.append(split_result['metrics'])
            
            # Collect individual predictions with split info
            for pred in split_result['results']:
                pred_copy = pred.copy()
                pred_copy['split'] = split_result['split']
                all_predictions.append(pred_copy)
        
        # Calculate aggregate statistics
        metrics_df = pd.DataFrame([{
            'split': i+1,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'], 
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'total_samples': metrics['total']
        } for i, metrics in enumerate(split_metrics)])
        
        # Calculate mean and std across splits
        aggregate_stats = {
            'accuracy_mean': metrics_df['accuracy'].mean(),
            'accuracy_std': metrics_df['accuracy'].std(),
            'f1_score_mean': metrics_df['f1_score'].mean(),
            'f1_score_std': metrics_df['f1_score'].std(),
            'precision_mean': metrics_df['precision'].mean(),
            'precision_std': metrics_df['precision'].std(),
            'recall_mean': metrics_df['recall'].mean(),
            'recall_std': metrics_df['recall'].std(),
            'total_samples': metrics_df['total_samples'].sum(),
            'n_splits': len(split_metrics)
        }
        
        aggregated_results[exp_name] = {
            'split_metrics': metrics_df,
            'aggregate_stats': aggregate_stats,
            'all_predictions': all_predictions,
            'individual_splits': exp_splits
        }
    
    return aggregated_results

def print_cross_split_analysis(aggregated_results):
    """Print comprehensive cross-split analysis"""
    
    print("\n" + "="*80)
    print("CROSS-SPLIT BRS PERFORMANCE ANALYSIS")
    print("="*80)
    print("Analysis across 5 splits for robust performance evaluation")
    
    # Create summary table
    summary_data = []
    for exp_name, exp_data in aggregated_results.items():
        stats = exp_data['aggregate_stats']
        summary_data.append([
            exp_name.split('_')[-1] if 'tuning' in exp_name else exp_name,  # Shortened name
            f"{stats['accuracy_mean']:.3f}±{stats['accuracy_std']:.3f}",
            f"{stats['f1_score_mean']:.3f}±{stats['f1_score_std']:.3f}",
            f"{stats['precision_mean']:.3f}±{stats['precision_std']:.3f}",
            f"{stats['recall_mean']:.3f}±{stats['recall_std']:.3f}",
            f"{stats['total_samples']}",
            f"{stats['n_splits']}"
        ])
    
    summary_df = pd.DataFrame(summary_data, columns=[
        'Model', 'Accuracy (Mean±Std)', 'F1 Score (Mean±Std)', 
        'Precision (Mean±Std)', 'Recall (Mean±Std)', 'Total Samples', 'Splits'
    ])
    
    print("\nCROSS-SPLIT PERFORMANCE SUMMARY:")
    print(summary_df.to_string(index=False))
    
    # Detailed analysis for each experiment
    print("\n" + "-"*80)
    print("DETAILED PER-EXPERIMENT ANALYSIS:")
    print("-"*80)
    
    for exp_name, exp_data in aggregated_results.items():
        print(f"\n{exp_name}:")
        print("-" * 50)
        
        split_df = exp_data['split_metrics']
        stats = exp_data['aggregate_stats']
        
        print("Per-split performance:")
        for _, row in split_df.iterrows():
            print(f"  Split {int(row['split'])}: Acc={row['accuracy']:.3f}, "
                  f"F1={row['f1_score']:.3f}, Prec={row['precision']:.3f}, Rec={row['recall']:.3f}")
        
        print(f"\nAggregate Statistics:")
        print(f"  Accuracy:  {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
        print(f"  F1 Score:  {stats['f1_score_mean']:.3f} ± {stats['f1_score_std']:.3f}")
        print(f"  Precision: {stats['precision_mean']:.3f} ± {stats['precision_std']:.3f}")
        print(f"  Recall:    {stats['recall_mean']:.3f} ± {stats['recall_std']:.3f}")
        
        # Variance analysis (coefficient of variation)
        cv_accuracy = stats['accuracy_std'] / stats['accuracy_mean'] if stats['accuracy_mean'] > 0 else 0
        cv_f1 = stats['f1_score_std'] / stats['f1_score_mean'] if stats['f1_score_mean'] > 0 else 0
        
        print(f"\nRobustness Analysis (Coefficient of Variation - lower is more robust):")
        print(f"  Accuracy CV:  {cv_accuracy:.3f}")
        print(f"  F1 Score CV:  {cv_f1:.3f}")

def identify_best_models(aggregated_results):
    """Identify best performing models and verify thesis statements"""
    
    print("\n" + "="*80)
    print("MODEL RANKING AND THESIS VERIFICATION")
    print("="*80)
    
    # Collect model performance for ranking
    model_performance = []
    
    for exp_name, exp_data in aggregated_results.items():
        stats = exp_data['aggregate_stats']
        
        # Create short model name
        if '017' in exp_name:
            model_name = 'Model_1 (M1)'
        elif '021' in exp_name:
            model_name = 'Model_2 (M2)'
        elif '026' in exp_name:
            model_name = 'Model_3 (M3)'
        else:
            model_name = exp_name
        
        model_performance.append({
            'name': model_name,
            'exp_name': exp_name,
            'accuracy_mean': stats['accuracy_mean'],
            'accuracy_std': stats['accuracy_std'],
            'f1_mean': stats['f1_score_mean'], 
            'f1_std': stats['f1_score_std'],
            'cv_accuracy': stats['accuracy_std'] / stats['accuracy_mean'] if stats['accuracy_mean'] > 0 else 0,
            'cv_f1': stats['f1_score_std'] / stats['f1_score_mean'] if stats['f1_score_mean'] > 0 else 0
        })
    
    # Sort by different criteria
    accuracy_ranking = sorted(model_performance, key=lambda x: x['accuracy_mean'], reverse=True)
    f1_ranking = sorted(model_performance, key=lambda x: x['f1_mean'], reverse=True)
    robustness_accuracy = sorted(model_performance, key=lambda x: x['cv_accuracy'])
    robustness_f1 = sorted(model_performance, key=lambda x: x['cv_f1'])
    
    print("\nRANKINGS:")
    print("-" * 50)
    
    print("By Mean Accuracy:")
    for i, model in enumerate(accuracy_ranking, 1):
        print(f"  {i}. {model['name']}: {model['accuracy_mean']:.3f}±{model['accuracy_std']:.3f}")
    
    print("\nBy Mean F1 Score:")
    for i, model in enumerate(f1_ranking, 1):
        print(f"  {i}. {model['name']}: {model['f1_mean']:.3f}±{model['f1_std']:.3f}")
    
    print("\nBy Robustness (Accuracy CV - lower is better):")
    for i, model in enumerate(robustness_accuracy, 1):
        print(f"  {i}. {model['name']}: CV = {model['cv_accuracy']:.3f}")
    
    print("\nBy Robustness (F1 CV - lower is better):")
    for i, model in enumerate(robustness_f1, 1):
        print(f"  {i}. {model['name']}: CV = {model['cv_f1']:.3f}")
    
    # Thesis verification
    print("\n" + "="*60)
    print("THESIS STATEMENT VERIFICATION (5-SPLIT ANALYSIS):")
    print("="*60)
    
    print("\nYour original claims:")
    print("1. 'Model M1 showed the lowest variance'")
    print("2. 'Model M2 achieved the highest F1 score and accuracy'")
    print("3. 'Model M3 was specialized in predicting BRS1 samples correctly'")
    
    print("\nACTUAL RESULTS:")
    
    # Check variance claim
    most_robust_acc = robustness_accuracy[0]['name']
    most_robust_f1 = robustness_f1[0]['name']
    print(f"→ MOST ROBUST (lowest accuracy CV): {most_robust_acc}")
    print(f"→ MOST ROBUST (lowest F1 CV): {most_robust_f1}")
    
    # Check performance claim  
    best_accuracy = accuracy_ranking[0]['name']
    best_f1 = f1_ranking[0]['name']
    print(f"→ HIGHEST ACCURACY: {best_accuracy}")
    print(f"→ HIGHEST F1 SCORE: {best_f1}")
    
    print("\n🔧 RECOMMENDED CORRECTIONS:")
    print(f"- Robustness: '{most_robust_acc} showed the lowest variance'")
    print(f"- Performance: '{best_accuracy} achieved the highest accuracy, {best_f1} achieved the highest F1 score'")
    
    return model_performance

def create_visualizations(aggregated_results):
    """Create comprehensive visualization plots"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Split BRS Performance Analysis (5 Splits)', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    plot_data = []
    for exp_name, exp_data in aggregated_results.items():
        split_df = exp_data['split_metrics']
        
        # Create short model name
        if '017' in exp_name:
            model_name = 'M1'
        elif '021' in exp_name:
            model_name = 'M2' 
        elif '026' in exp_name:
            model_name = 'M3'
        else:
            model_name = exp_name
        
        for _, row in split_df.iterrows():
            plot_data.append({
                'Model': model_name,
                'Split': f"Split {int(row['split'])}",
                'Accuracy': row['accuracy'],
                'F1_Score': row['f1_score'],
                'Precision': row['precision'],
                'Recall': row['recall']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Accuracy across splits
    sns.boxplot(data=plot_df, x='Model', y='Accuracy', ax=axes[0,0])
    axes[0,0].set_title('Accuracy Distribution Across Splits')
    axes[0,0].set_ylabel('Accuracy')
    
    # 2. F1 Score across splits
    sns.boxplot(data=plot_df, x='Model', y='F1_Score', ax=axes[0,1])
    axes[0,1].set_title('F1 Score Distribution Across Splits')
    axes[0,1].set_ylabel('F1 Score')
    
    # 3. Mean performance comparison
    mean_data = []
    for exp_name, exp_data in aggregated_results.items():
        stats = exp_data['aggregate_stats']
        if '017' in exp_name:
            model_name = 'M1'
        elif '021' in exp_name:
            model_name = 'M2'
        elif '026' in exp_name:
            model_name = 'M3'
        else:
            model_name = exp_name
            
        mean_data.append({
            'Model': model_name,
            'Accuracy': stats['accuracy_mean'],
            'F1_Score': stats['f1_score_mean'],
            'Accuracy_Std': stats['accuracy_std'],
            'F1_Std': stats['f1_score_std']
        })
    
    mean_df = pd.DataFrame(mean_data)
    
    x_pos = np.arange(len(mean_df))
    axes[1,0].bar(x_pos, mean_df['Accuracy'], yerr=mean_df['Accuracy_Std'], 
                  capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1,0].set_title('Mean Accuracy with Error Bars')
    axes[1,0].set_ylabel('Mean Accuracy ± Std')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(mean_df['Model'])
    
    axes[1,1].bar(x_pos, mean_df['F1_Score'], yerr=mean_df['F1_Std'],
                  capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1,1].set_title('Mean F1 Score with Error Bars')
    axes[1,1].set_ylabel('Mean F1 Score ± Std')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(mean_df['Model'])
    
    plt.tight_layout()
    
    # Save plots
    output_path = '/gris/gris-f/homelv/phempel/masterthesis/MMFL/'
    plt.savefig(os.path.join(output_path, 'cross_split_brs_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'cross_split_brs_analysis.pdf'), bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    
    # Configuration
    base_dir = '/gris/gris-f/homelv/phempel/masterthesis/test/'
    test_csv = None  # Set to CSV path if you want to filter test samples only
    
    # Experiment configurations (without split specification)
    experiments = [
        'output_advanced_hierarchical_tuning_12-16-32_0_017_splits',
        'output_advanced_hierarchical_tuning_12-16-32_0_021_splits', 
        'output_grid_search_hierarchical_vacation_18-12-49_0_026_splits'
    ]
    
    print("🔬 COMPREHENSIVE BRS ANALYSIS ACROSS ALL SPLITS")
    print("=" * 60)
    print(f"Analyzing {len(experiments)} models across 5 splits each...")
    
    # Collect results from all experiments and splits
    all_results = []
    
    for exp_name in experiments:
        print(f"\nProcessing experiment: {exp_name}")
        
        for split_num in range(1, 6):  # splits 1-5
            print(f"  Loading split {split_num}...", end=" ")
            
            result = run_single_experiment_analysis(base_dir, exp_name, split_num, test_csv)
            
            if result:
                all_results.append(result)
                print(f"✓ ({result['metrics']['total']} samples)")
            else:
                print("✗ (failed/missing)")
    
    if not all_results:
        print("❌ No results found! Check experiment names and directory paths.")
        return
    
    print(f"\n✅ Successfully loaded {len(all_results)} experiment-split combinations")
    
    # Aggregate results across splits
    print("\n📊 Aggregating results across splits...")
    aggregated_results = aggregate_cross_split_results(all_results)
    
    # Print comprehensive analysis
    print_cross_split_analysis(aggregated_results)
    
    # Identify best models and verify thesis
    model_rankings = identify_best_models(aggregated_results)
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    create_visualizations(aggregated_results)
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    output_path = '/gris/gris-f/homelv/phempel/masterthesis/MMFL/'
    
    # Save aggregate statistics
    aggregate_stats_list = []
    for exp_name, exp_data in aggregated_results.items():
        stats = exp_data['aggregate_stats']
        stats['experiment'] = exp_name
        aggregate_stats_list.append(stats)
    
    aggregate_df = pd.DataFrame(aggregate_stats_list)
    aggregate_df.to_csv(os.path.join(output_path, 'cross_split_aggregate_stats.csv'), index=False)
    
    # Save per-split details
    all_split_details = []
    for exp_name, exp_data in aggregated_results.items():
        split_df = exp_data['split_metrics'].copy()
        split_df['experiment'] = exp_name
        all_split_details.append(split_df)
    
    combined_splits_df = pd.concat(all_split_details, ignore_index=True)
    combined_splits_df.to_csv(os.path.join(output_path, 'cross_split_detailed_results.csv'), index=False)
    
    print("\n✅ Analysis complete!")
    print("Files saved:")
    print("- cross_split_brs_analysis.png/pdf (visualizations)")
    print("- cross_split_aggregate_stats.csv (aggregate statistics)")
    print("- cross_split_detailed_results.csv (per-split details)")
    
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print("Based on 5-split cross-validation, update your thesis statements according to the verification results above.")

if __name__ == "__main__":
    main()