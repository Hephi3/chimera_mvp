from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json

ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results"
OLD_DATA_ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results_old_data"

# CF Solo Baseline experiments for drift calculation
CF_SOLO_BASELINE_EXPERIMENTS = [
    "73_CF_Solo_sp1_s3",
    "73_CF_Solo_sp2_s3",
    "73_CF_Solo_sp3_s3",
    "73_CF_Solo_sp4_s3",
    "73_CF_Solo_sp5_s3"
]

# CF Solo Baseline with all seeds (s1, s2, s3)
CF_SOLO_BASELINE_ALL_SEEDS = [
    "73_CF_Solo_sp1_s1",
    "73_CF_Solo_sp2_s1",
    "73_CF_Solo_sp3_s1",
    "73_CF_Solo_sp4_s1",
    "73_CF_Solo_sp5_s1",
    "73_CF_Solo_sp1_s2",
    "73_CF_Solo_sp2_s2",
    "73_CF_Solo_sp3_s2",
    "73_CF_Solo_sp4_s2",
    "73_CF_Solo_sp5_s2",
    "73_CF_Solo_sp1_s3",
    "73_CF_Solo_sp2_s3",
    "73_CF_Solo_sp3_s3",
    "73_CF_Solo_sp4_s3",
    "73_CF_Solo_sp5_s3"
]

# Test metrics only
test_metrics = {
    # 'Accuracy/test': 'Test Accuracy',
    # 'Binary_Accuracy/test': 'Test Binary Accuracy',
    # 'ROC_AUC/test': 'Test ROC AUC',
    'F1/test': 'F1 Score',
    'Avg_f1/test': 'PCF',
    'Client_Drift/test': 'Client Drift',
}


def tensorboard_to_datadict_federated(experiment_name: str, fold_num: int, exp_dir: str = ROOT_RESULTS):
    """Extract server test data from TensorBoard logs for a specific fold"""
    
    experiment_dir = os.path.join(exp_dir, experiment_name, f"Fold{fold_num}")
    log_dir = os.path.join(experiment_dir, "log")
    
    if not os.path.exists(log_dir):
        if exp_dir == ROOT_RESULTS:
            old_experiment_dir = os.path.join(OLD_DATA_ROOT_RESULTS, experiment_name, f"Fold{fold_num}")
            old_log_dir = os.path.join(old_experiment_dir, "log")
            if os.path.exists(old_log_dir):
                log_dir = old_log_dir
            else:
                print(f"Warning: Log directory {log_dir} does not exist for fold {fold_num}")
                return {}
        else:
            print(f"Warning: Log directory {log_dir} does not exist for fold {fold_num}")
            return {}
    
    server_data = defaultdict(lambda: defaultdict(dict))
    
    for item in os.listdir(log_dir):
        if "server" in item:
            parts = item.split("_")
            round_num = int(parts[-1])
            
            server_path = os.path.join(log_dir, item)
            if not os.path.exists(server_path):
                continue
                
            for file in os.listdir(server_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(server_path, file)
                    try:
                        for event in tf.compat.v1.train.summary_iterator(event_file):
                            for value in event.summary.value:
                                server_data[round_num][value.tag] = value.simple_value
                    except Exception as e:
                        print(f"Warning: Could not read {event_file}: {e}")
                        continue
    
    return dict(server_data)


def get_available_folds(experiment_name: str, exp_dir: str = ROOT_RESULTS):
    """Get list of available fold numbers for an experiment"""
    experiment_dir = os.path.join(exp_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
        if exp_dir == ROOT_RESULTS:
            old_experiment_dir = os.path.join(OLD_DATA_ROOT_RESULTS, experiment_name)
            if os.path.exists(old_experiment_dir):
                experiment_dir = old_experiment_dir
            else:
                raise ValueError(f"Experiment directory {experiment_dir} does not exist")
        else:
            raise ValueError(f"Experiment directory {experiment_dir} does not exist")
    
    folds = []
    for item in os.listdir(experiment_dir):
        if item.startswith("Fold") and os.path.isdir(os.path.join(experiment_dir, item)):
            try:
                fold_num = int(item[4:])
                folds.append(fold_num)
            except ValueError:
                continue
    
    return sorted(folds)


def compute_crossfold_experiment_average(experiment_name, fold_numbers, metric, all_rounds):
    """Compute cross-validation average for a single experiment across its folds"""
    
    fold_data = []
    for fold_num in fold_numbers:
        try:
            server_data = tensorboard_to_datadict_federated(experiment_name, fold_num)
            fold_data.append(server_data)
        except Exception as e:
            print(f"Warning: Could not load data for experiment {experiment_name} fold {fold_num}: {e}")
            continue
    
    if not fold_data:
        return None
    
    server_cv_data = {}
    for round_num in all_rounds:
        round_values = []
        for server_data in fold_data:
            if round_num in server_data and metric in server_data[round_num]:
                round_values.append(server_data[round_num][metric])
        
        if round_values:
            server_cv_data[round_num] = np.mean(round_values)
    
    return server_cv_data


def compute_crossfold_group_statistics(experiment_group, fold_numbers, metric, all_rounds):
    """Compute statistics across multiple cross-validation experiments in a group"""
    
    experiment_cv_averages = []
    
    for exp_name in experiment_group:
        server_cv_data = compute_crossfold_experiment_average(exp_name, fold_numbers, metric, all_rounds)
        
        if server_cv_data is not None:
            experiment_cv_averages.append(server_cv_data)
    
    if not experiment_cv_averages:
        return None
    
    server_stats = {}
    for round_num in all_rounds:
        experiment_round_values = []
        
        for server_cv_data in experiment_cv_averages:
            if round_num in server_cv_data:
                experiment_round_values.append(server_cv_data[round_num])
        
        if experiment_round_values:
            server_stats[round_num] = {
                'mean': np.mean(experiment_round_values),
                'std': np.std(experiment_round_values, ddof=1) if len(experiment_round_values) > 1 else 0.0
            }
    
    return server_stats


def load_baseline_f1_stats(fold_numbers, submodel, all_rounds, all_seeds=False):
    """Load CF Solo Baseline F1 statistics for drift calculation"""
    
    baseline_experiments = CF_SOLO_BASELINE_ALL_SEEDS if all_seeds else CF_SOLO_BASELINE_EXPERIMENTS
    baseline_name = "all seeds" if all_seeds else "seed 3 only"
    print(f"Loading CF Solo Baseline ({baseline_name}) for drift calculation...")
    
    # Compute F1 stats for stage 0 and stage 1
    metric_stage_0 = f"F1/test/{submodel}/0"
    metric_stage_1 = f"F1/test/{submodel}/1"
    
    baseline_stats_0 = compute_crossfold_group_statistics(
        baseline_experiments, fold_numbers, metric_stage_0, all_rounds)
    
    baseline_stats_1 = compute_crossfold_group_statistics(
        baseline_experiments, fold_numbers, metric_stage_1, all_rounds)
    
    return baseline_stats_0, baseline_stats_1


def compute_client_drift(experiment_f1_stage_0, experiment_f1_stage_1, 
                        baseline_f1_stage_0, baseline_f1_stage_1, all_rounds):
    """
    Compute client drift as: (Experiment F1 - Baseline F1) averaged across both stages
    Negative values indicate the experiment performs worse than baseline (drift detected)
    """
    
    drift_stats = {}
    
    for round_num in all_rounds:
        # Get F1 values for both stages from experiment
        exp_f1_s0 = experiment_f1_stage_0.get(round_num, {}).get('mean', None) if experiment_f1_stage_0 else None
        exp_f1_s1 = experiment_f1_stage_1.get(round_num, {}).get('mean', None) if experiment_f1_stage_1 else None
        
        # Get F1 values for both stages from baseline
        base_f1_s0 = baseline_f1_stage_0.get(round_num, {}).get('mean', None) if baseline_f1_stage_0 else None
        base_f1_s1 = baseline_f1_stage_1.get(round_num, {}).get('mean', None) if baseline_f1_stage_1 else None
        
        # Calculate drift for each stage
        drifts = []
        if exp_f1_s0 is not None and base_f1_s0 is not None:
            drift_s0 = exp_f1_s0 - base_f1_s0
            drifts.append(drift_s0)
        
        if exp_f1_s1 is not None and base_f1_s1 is not None:
            drift_s1 = exp_f1_s1 - base_f1_s1
            drifts.append(drift_s1)
        
        # Average the drifts
        if drifts:
            drift_stats[round_num] = {
                'mean': np.mean(drifts),
                'std': 0.0  # We don't compute std for drift metric
            }
    
    return drift_stats


def plot_test_metric(ax, title, server_stats, all_rounds, show_std=True, 
                     show_legend=True, color='blue', label_prefix=''):
    """Plot test metrics with stage differentiation"""
    
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.12
    
    if not server_stats:
        return
    
    stage_0_rounds, stage_1_rounds = [], []
    stage_0_means, stage_1_means = [], []
    stage_0_stds, stage_1_stds = [], []
    
    for round_num in sorted(server_stats.keys()):
        if round_num in all_rounds:
            round_data = server_stats[round_num]
            
            if 'stage_0' in round_data:
                stage_0_rounds.append(round_num)
                stage_0_means.append(round_data['stage_0']['mean'])
                stage_0_stds.append(round_data['stage_0']['std'])
            
            if 'stage_1' in round_data:
                stage_1_rounds.append(round_num)
                stage_1_means.append(round_data['stage_1']['mean'])
                stage_1_stds.append(round_data['stage_1']['std'])
    
    if stage_0_means:
        label = f'{label_prefix}' if label_prefix else 'Stage 1'
        ax.plot(stage_0_rounds, stage_0_means, color=color, linewidth=2, markersize=6, 
               label=label, alpha=LINE_ALPHA, linestyle='-')
        
        if show_std and any(std > 0 for std in stage_0_stds):
            stage_0_means_arr = np.array(stage_0_means)
            stage_0_stds_arr = np.array(stage_0_stds)
            ax.fill_between(stage_0_rounds, stage_0_means_arr - stage_0_stds_arr, 
                           stage_0_means_arr + stage_0_stds_arr, color=color, alpha=STD_ALPHA)
    
    if stage_1_means:
        # Don't add label if stage_0 already added one for this group
        label = '' if (label_prefix and stage_0_means) else 'Stage 2'
        ax.plot(stage_1_rounds, stage_1_means, color=color, linewidth=2, markersize=6, 
               label=label, alpha=LINE_ALPHA, linestyle='--')
        
        if show_std and any(std > 0 for std in stage_1_stds):
            stage_1_means_arr = np.array(stage_1_means)
            stage_1_stds_arr = np.array(stage_1_stds)
            ax.fill_between(stage_1_rounds, stage_1_means_arr - stage_1_stds_arr, 
                           stage_1_means_arr + stage_1_stds_arr, color=color, alpha=STD_ALPHA)
    
    ax.set_xlabel('Federated Round')
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def plot_avg_f1_metric(ax, title, server_stats, all_rounds, show_std=True, 
                       show_legend=True, color='blue', label_prefix=''):
    """Plot Avg_f1 metric"""
    
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.12

    if not server_stats:
        return

    server_rounds, server_means, server_stds = [], [], []

    for round_num in sorted(server_stats.keys()):
        if round_num in all_rounds:
            server_rounds.append(round_num)
            server_means.append(server_stats[round_num].get('mean', 0.0))
            server_stds.append(server_stats[round_num].get('std', 0.0))

    if not server_rounds:
        return

    label = f'{label_prefix}' if label_prefix else 'Avg F1'
    ax.plot(server_rounds, server_means, color=color, linewidth=2, markersize=6,
            label=label, alpha=LINE_ALPHA)

    if show_std and any(std > 0 for std in server_stds):
        server_means_arr = np.array(server_means)
        server_stds_arr = np.array(server_stds)
        ax.fill_between(server_rounds, server_means_arr - server_stds_arr,
                        server_means_arr + server_stds_arr, color=color, alpha=STD_ALPHA)

    ax.set_xlabel('Federated Round')
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def plot_drift_metric(ax, title, drift_stats, all_rounds, show_std=True, 
                      show_legend=True, color='blue', label_prefix=''):
    """Plot client drift metric"""
    
    LINE_ALPHA = 0.8
    
    if not drift_stats:
        return
    
    drift_rounds, drift_means = [], []
    
    for round_num in sorted(drift_stats.keys()):
        if round_num in all_rounds:
            drift_rounds.append(round_num)
            drift_means.append(drift_stats[round_num].get('mean', 0.0))
    
    if not drift_rounds:
        return
    
    label = f'{label_prefix}' if label_prefix else 'Client Drift'
    ax.plot(drift_rounds, drift_means, color=color, linewidth=2, markersize=6,
            label=label, alpha=LINE_ALPHA)
    
    # Add horizontal line at y=0 to show no-drift reference
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Federated Round')
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def expand_experiment_names(experiment_groups: dict) -> dict:
    """Expand experiment names to include all split and seed combinations.
    
    For each base experiment name, generates all combinations of:
    - 5 splits: sp1, sp2, sp3, sp4, sp5
    - 3 seeds: s1, s2, s3
    
    Validates that all generated experiment directories exist, raising an error if any are missing.
    
    Example: "art_CFCDID_no_weighted_training" -> 
             ["art_CFCDID_no_weighted_training_sp1_s1", ..., "art_CFCDID_no_weighted_training_sp5_s3"]
    """
    expanded_groups = {}
    
    for group_name, experiment_list in experiment_groups.items():
        expanded_experiments = []
        missing_experiments = []
        
        for base_exp_name in experiment_list:
            # Generate all combinations of splits and seeds
            for split in range(1, 6):  # sp1 to sp5
                for seed in range(1, 4):  # s1 to s3
                    expanded_name = f"{base_exp_name}_sp{split}_s{seed}"
                    
                    # Check if experiment exists
                    try:
                        get_available_folds(expanded_name)
                        expanded_experiments.append(expanded_name)
                    except ValueError:
                        missing_experiments.append(expanded_name)
        
        if missing_experiments:
            raise ValueError(
                f"Error in group '{group_name}': The following experiments do not exist:\n" +
                "\n".join(f"  - {exp}" for exp in missing_experiments) +
                f"\n\nAll 15 combinations (5 splits × 3 seeds) must exist when using --gen_all_seeds."
            )
        
        expanded_groups[group_name] = expanded_experiments
    
    return expanded_groups


def plot_test_metrics_comparison(experiment_groups: dict, submodel: str = 'MM',
                                 show_std: bool = True, folds: list = None, name: str = 'test_metrics',
                                 all_seeds: bool = False):
    """Compare test metrics across groups of cross-fold validation experiments"""
    
    group_colors = ['red', 'green', 'purple', 'orange',  'brown', 'pink', 'gray']
    
    # Prepare experiment groups
    all_group_data = {}
    for group_name, experiment_list in experiment_groups.items():
        if folds is None:
            available_folds = get_available_folds(experiment_list[0])
            if not available_folds:
                print(f"Warning: No folds found for experiments in group {group_name}")
                continue
            fold_numbers = available_folds
        else:
            fold_numbers = folds
        
        valid_experiments = []
        for exp_name in experiment_list:
            exp_folds = get_available_folds(exp_name)
            if all(fold in exp_folds for fold in fold_numbers):
                valid_experiments.append(exp_name)
        
        if valid_experiments:
            all_group_data[group_name] = {'experiments': valid_experiments, 'folds': fold_numbers}
    
    if not all_group_data:
        raise ValueError("No valid experiment groups found")
    
    # Get all rounds
    global_all_rounds = set()
    for group_name, group_data in all_group_data.items():
        for exp_name in group_data['experiments']:
            for fold_num in group_data['folds']:
                try:
                    server_data = tensorboard_to_datadict_federated(exp_name, fold_num)
                    global_all_rounds.update(server_data.keys())
                except:
                    continue
    
    global_all_rounds = sorted(global_all_rounds)
    
    # Load baseline F1 stats for drift calculation
    # Use the first group's folds (assuming all groups use same folds)
    first_group_folds = list(all_group_data.values())[0]['folds']
    baseline_f1_stage_0, baseline_f1_stage_1 = load_baseline_f1_stats(
        first_group_folds, submodel, global_all_rounds, all_seeds)
    
    # Create plot - dynamic layout based on number of metrics
    num_metrics = len(test_metrics)
    ncols = 2
    nrows = (num_metrics + ncols - 1) // ncols  # ceiling division
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, 4 * nrows))
    axs = axs.flatten()  # flatten for easy indexing
    
    group_names_str = " vs ".join(all_group_data.keys())
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Compute statistics for all groups
    all_group_stats = {}
    for group_name, group_data in all_group_data.items():
        print(f"Computing statistics for: {group_name}")
        all_group_stats[group_name] = {}
        
        for base_metric in test_metrics.keys():
            if base_metric == 'Avg_f1/test' or base_metric == 'Client_Drift/test':
                continue
            
            metric_stage_0 = f"{base_metric}/{submodel}/0"
            metric_stage_1 = f"{base_metric}/{submodel}/1"
            
            server_stats_0 = compute_crossfold_group_statistics(
                group_data['experiments'], group_data['folds'], metric_stage_0, global_all_rounds)
            
            server_stats_1 = compute_crossfold_group_statistics(
                group_data['experiments'], group_data['folds'], metric_stage_1, global_all_rounds)
            
            # Combine stages
            combined_server_stats = {}
            if server_stats_0:
                for round_num in server_stats_0:
                    if round_num not in combined_server_stats:
                        combined_server_stats[round_num] = {}
                    combined_server_stats[round_num]['stage_0'] = server_stats_0[round_num]
            if server_stats_1:
                for round_num in server_stats_1:
                    if round_num not in combined_server_stats:
                        combined_server_stats[round_num] = {}
                    combined_server_stats[round_num]['stage_1'] = server_stats_1[round_num]
            
            if base_metric == 'F1/test':
                all_group_stats[group_name]['_f1_stage_0'] = server_stats_0
                all_group_stats[group_name]['_f1_stage_1'] = server_stats_1
            
            all_group_stats[group_name][base_metric] = combined_server_stats
    
    # Compute avg_f1 for each group
    for group_name in all_group_stats.keys():
        if '_f1_stage_0' in all_group_stats[group_name] and '_f1_stage_1' in all_group_stats[group_name]:
            f1_stage_0 = all_group_stats[group_name]['_f1_stage_0']
            f1_stage_1 = all_group_stats[group_name]['_f1_stage_1']
            
            avg_f1_stats = {}
            all_rounds_sorted = sorted(global_all_rounds)
            midpoint = len(all_rounds_sorted) // 2
            
            if midpoint > 0 and midpoint < len(all_rounds_sorted):
                last_stage1_round = all_rounds_sorted[midpoint]
                
                f1_s1test_s1_baseline = f1_stage_0.get(last_stage1_round, {}).get('mean', 0)
                f1_s2test_s1_baseline = f1_stage_1.get(last_stage1_round, {}).get('mean', 0)
                
                for round_num in all_rounds_sorted[midpoint:]:
                    f1_s1test_s2 = f1_stage_0.get(round_num, {}).get('mean', None)
                    f1_s2test_s2 = f1_stage_1.get(round_num, {}).get('mean', None)
                    
                    if f1_s1test_s2 is not None and f1_s2test_s2 is not None:
                        avg_f1 = 0.5 * f1_s1test_s2 + 0.5 * f1_s2test_s2 - (0.5 * f1_s1test_s1_baseline + 0.5 * f1_s2test_s1_baseline)
                        avg_f1_stats[round_num] = {'mean': avg_f1, 'std': 0.0}
            
            all_group_stats[group_name]['Avg_f1/test'] = avg_f1_stats
    
    # Compute client drift for each group
    for group_name in all_group_stats.keys():
        if '_f1_stage_0' in all_group_stats[group_name] and '_f1_stage_1' in all_group_stats[group_name]:
            exp_f1_stage_0 = all_group_stats[group_name]['_f1_stage_0']
            exp_f1_stage_1 = all_group_stats[group_name]['_f1_stage_1']
            
            drift_stats = compute_client_drift(
                exp_f1_stage_0, exp_f1_stage_1,
                baseline_f1_stage_0, baseline_f1_stage_1,
                global_all_rounds
            )
            
            all_group_stats[group_name]['Client_Drift/test'] = drift_stats
    
    # Calculate midpoint for stage transition line
    all_rounds_sorted = sorted(global_all_rounds)
    midpoint = len(all_rounds_sorted) // 2
    stage_transition_round = all_rounds_sorted[midpoint] if midpoint < len(all_rounds_sorted) else None
    
    # Plot each metric
    for i, (base_metric, title) in enumerate(test_metrics.items()):
        ax = axs[i]
        
        for group_idx, (group_name, group_data) in enumerate(all_group_data.items()):
            if group_name not in all_group_stats or base_metric not in all_group_stats[group_name]:
                continue
                
            server_stats = all_group_stats[group_name][base_metric]
            color = group_colors[group_idx % len(group_colors)]
            
            # Don't show legend on individual plots
            show_legend_here = False
            
            if 'Avg_f1' in base_metric:
                plot_avg_f1_metric(ax, title, server_stats, global_all_rounds,
                                 show_std=show_std, show_legend=show_legend_here, color=color,
                                 label_prefix=group_name)
            elif 'Client_Drift' in base_metric:
                plot_drift_metric(ax, title, server_stats, global_all_rounds,
                                show_std=show_std, show_legend=show_legend_here, color=color,
                                label_prefix=group_name)
            else:
                plot_test_metric(ax, title, server_stats, global_all_rounds,
                               show_std=show_std, show_legend=show_legend_here, color=color,
                               label_prefix=group_name)
        
        # Add vertical line at stage transition (not for drift metric)
        if stage_transition_round is not None and 'Client_Drift' not in base_metric:
            ax.axvline(x=stage_transition_round, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Hide any extra subplots
    for idx in range(num_metrics, len(axs)):
        axs[idx].axis('off')
    
    # Collect legend elements
    legend_handles = []
    legend_labels = []
    
    # Add group entries
    for group_idx, group_name in enumerate(all_group_data.keys()):
        color = group_colors[group_idx % len(group_colors)]
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=group_name))
        legend_labels.append(group_name)
    
    # Add stage line style indicators
    legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Stage 1'))
    legend_labels.append('Stage 1')
    legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Stage 2'))
    legend_labels.append('Stage 2')
    
    # For 3 metrics (with empty bottom-right spot), place legend there
    # Otherwise place it below the entire plot
    if num_metrics == 3:
        # Place legend in the bottom-right empty subplot
        axs[3].legend(handles=legend_handles, labels=legend_labels, 
                     loc='center', fontsize=10, frameon=True)
        plt.tight_layout()
    else:
        # Create legend below the entire plot as a horizontal line
        fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center', 
                   ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.05), 
                   fontsize=10, frameon=True)
        plt.tight_layout()
    
    plot_path = os.path.join(ROOT_RESULTS, f"{name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Metrics Comparison with Client Drift')
    parser.add_argument('--groups', type=str, required=True, help='JSON string defining experiment groups')
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD'])
    parser.add_argument('--show_std', action='store_true', default=True)
    parser.add_argument('--no_std', action='store_true')
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--name', type=str, default='test_metrics')
    parser.add_argument('--all_seeds', action='store_true', help='Use all seeds (s1, s2, s3) for baseline instead of just s3')
    parser.add_argument('--gen_all_seeds', action='store_true', help='Automatically expand experiment names to all split/seed combinations (sp1-sp5, s1-s3)')
    
    args = parser.parse_args()
    
    try:
        experiment_groups = json.loads(args.groups)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for --groups argument")
        exit(1)
    
    # Expand experiment names if --gen_all_seeds is specified
    if args.gen_all_seeds:
        print("Expanding experiment names to all split/seed combinations...")
        experiment_groups = expand_experiment_names(experiment_groups)
    
    show_std = args.show_std and not args.no_std
    
    plot_test_metrics_comparison(experiment_groups, args.submodel, show_std, args.folds, args.name, args.all_seeds)
