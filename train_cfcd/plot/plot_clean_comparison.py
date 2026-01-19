from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json

ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results"
OLD_DATA_ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results_old_data"


def tensorboard_to_datadict_federated(experiment_name: str, fold_num: int, exp_dir: str = ROOT_RESULTS):
    """Extract data from TensorBoard logs for a specific fold"""
    
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
    
    data = defaultdict(lambda: defaultdict(dict))
    
    # Load server data (test metrics)
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
                                data[round_num][value.tag] = value.simple_value
                    except Exception as e:
                        print(f"Warning: Could not read {event_file}: {e}")
                        continue
    
    # Load client data (training metrics)
    for item in os.listdir(log_dir):
        if "client" in item:
            parts = item.split("_")
            client_id = int(parts[1])
            round_num = int(parts[-1])
            
            client_path = os.path.join(log_dir, item)
            if not os.path.exists(client_path):
                continue
                
            for file in os.listdir(client_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(client_path, file)
                    try:
                        for event in tf.compat.v1.train.summary_iterator(event_file):
                            for value in event.summary.value:
                                # Store client metrics
                                if round_num not in data:
                                    data[round_num] = {}
                                if f'client_{client_id}_{value.tag}' not in data[round_num]:
                                    data[round_num][f'client_{client_id}_{value.tag}'] = value.simple_value
                    except Exception as e:
                        print(f"Warning: Could not read {event_file}: {e}")
                        continue
    
    return dict(data)


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


def compute_crossfold_experiment_average(experiment_name, fold_numbers, metric, debug=False):
    """Compute cross-validation average for a single experiment across its folds"""
    
    fold_data = []
    for fold_num in fold_numbers:
        try:
            data = tensorboard_to_datadict_federated(experiment_name, fold_num)
            fold_data.append(data)
            
            # Debug: print available keys
            if debug and fold_num == fold_numbers[0]:
                print(f"\nDebug - Available keys in {experiment_name} Fold{fold_num}, Round 1:")
                if 1 in data:
                    for key in sorted(data[1].keys()):
                        if 'Loss' in key or 'loss' in key:
                            print(f"  {key}: {data[1][key]}")
        except Exception as e:
            print(f"Warning: Could not load data for experiment {experiment_name} fold {fold_num}: {e}")
            continue
    
    if not fold_data:
        return None
    
    cv_data = {}
    all_rounds = set()
    for data in fold_data:
        all_rounds.update(data.keys())
    
    for round_num in sorted(all_rounds):
        round_values = []
        for data in fold_data:
            if round_num in data:
                # Check for direct metric
                if metric in data[round_num]:
                    round_values.append(data[round_num][metric])
                # For training/validation loss, aggregate across clients
                elif '/train/' in metric or '/val/' in metric:
                    client_values = []
                    for key, value in data[round_num].items():
                        if key.startswith('client_') and metric in key:
                            client_values.append(value)
                    if client_values:
                        round_values.append(np.mean(client_values))
        
        if round_values:
            cv_data[round_num] = np.mean(round_values)
    
    return cv_data


def compute_crossfold_group_statistics(experiment_group, fold_numbers, metric, debug=False):
    """Compute statistics across multiple cross-validation experiments in a group"""
    
    experiment_cv_averages = []
    
    for idx, exp_name in enumerate(experiment_group):
        cv_data = compute_crossfold_experiment_average(exp_name, fold_numbers, metric, debug=(debug and idx==0))
        
        if cv_data is not None:
            experiment_cv_averages.append(cv_data)
    
    if not experiment_cv_averages:
        return None
    
    stats = {}
    all_rounds = set()
    for cv_data in experiment_cv_averages:
        all_rounds.update(cv_data.keys())
    
    for round_num in sorted(all_rounds):
        experiment_round_values = []
        
        for cv_data in experiment_cv_averages:
            if round_num in cv_data:
                experiment_round_values.append(cv_data[round_num])
        
        if experiment_round_values:
            stats[round_num] = {
                'mean': np.mean(experiment_round_values),
                'std': np.std(experiment_round_values, ddof=1) if len(experiment_round_values) > 1 else 0.0
            }
    
    return stats


def smooth_data(rounds, values, window=5):
    """Apply moving average smoothing"""
    if window <= 1 or len(values) < window:
        return rounds, values
    
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    start_offset = (window - 1) // 2
    smoothed_rounds = rounds[start_offset:start_offset + len(smoothed)]
    
    return smoothed_rounds, smoothed


def plot_metric(ax, title, group_stats, show_std=True, colors=None, group_names=None, 
                smooth_window=1, ylabel=None):
    """Plot a single metric with group comparison"""
    
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.12
    
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for group_idx, (group_name, stats) in enumerate(group_stats.items()):
        if not stats:
            continue
        
        rounds = sorted(stats.keys())
        means = [stats[r]['mean'] for r in rounds]
        stds = [stats[r]['std'] for r in rounds]
        
        # Apply smoothing if requested
        if smooth_window > 1:
            rounds, means = smooth_data(rounds, means, smooth_window)
            _, stds = smooth_data(sorted(stats.keys()), 
                                 [stats[r]['std'] for r in sorted(stats.keys())], 
                                 smooth_window)
        
        color = colors[group_idx % len(colors)]
        label = group_names[group_idx] if group_names else group_name
        
        ax.plot(rounds, means, color=color, linewidth=2, label=label, alpha=LINE_ALPHA)
        
        if show_std and any(std > 0 for std in stds):
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(rounds, means_arr - stds_arr, means_arr + stds_arr, 
                           color=color, alpha=STD_ALPHA)
    
    ax.set_xlabel('Federated Round', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel if ylabel else title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def plot_comparison(experiment_groups: dict, submodel: str = 'MM', show_std: bool = True, 
                   folds: list = None, name: str = 'comparison', smooth_window: int = 1):
    """Compare Training Loss and Test F1 Score across experiment groups"""
    
    group_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
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
    
    # Create figure with 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Define metrics to plot
    metrics = {
        f'Loss/train/{submodel}': ('Training Loss', 'Loss'),
        f'F1/test/{submodel}/0': ('Test F1 Score', 'F1 Score'),
    }
    
    # Compute statistics for all groups
    all_group_stats = {metric: {} for metric in metrics.keys()}
    
    for group_idx, (group_name, group_data) in enumerate(all_group_data.items()):
        print(f"Computing statistics for: {group_name}")
        
        for metric_key in metrics.keys():
            stats = compute_crossfold_group_statistics(
                group_data['experiments'], group_data['folds'], metric_key, debug=(group_idx==0))
            
            if stats:
                all_group_stats[metric_key][group_name] = stats
    
    # Plot each metric
    for ax, (metric_key, (title, ylabel)) in zip(axs, metrics.items()):
        plot_metric(ax, title, all_group_stats[metric_key], 
                   show_std=show_std, colors=group_colors, 
                   group_names=list(all_group_data.keys()),
                   smooth_window=smooth_window, ylabel=ylabel)
    
    # Create horizontal legend below the plots
    legend_handles = []
    legend_labels = []
    
    for group_idx, group_name in enumerate(all_group_data.keys()):
        color = group_colors[group_idx % len(group_colors)]
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        legend_labels.append(group_name)
    
    fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center', 
               ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.1), 
               fontsize=12, frameon=True)
    
    plt.tight_layout()
    
    plot_path = os.path.join(ROOT_RESULTS, f"{name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean comparison of Training Loss and Test F1')
    parser.add_argument('--groups', type=str, required=True, 
                       help='JSON string defining experiment groups')
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD'])
    parser.add_argument('--show_std', action='store_true', default=True)
    parser.add_argument('--no_std', action='store_true')
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--name', type=str, default='comparison')
    parser.add_argument('--smooth_window', type=int, default=1,
                       help='Window size for moving average smoothing (default: 1, no smoothing)')
    
    args = parser.parse_args()
    
    try:
        experiment_groups = json.loads(args.groups)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for --groups argument")
        exit(1)
    
    show_std = args.show_std and not args.no_std
    
    plot_comparison(experiment_groups, args.submodel, show_std, args.folds, args.name, args.smooth_window)
