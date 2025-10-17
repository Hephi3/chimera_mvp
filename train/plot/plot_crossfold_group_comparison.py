import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/results"
OLD_DATA_ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/results_old_data"

def smooth_client_data(client_data, window_size=5):
    """
    Apply smoothing to federated client data structure.
    
    Args:
        client_data: {client_id: {round: {metric: [values]}}}
        window_size: Size of the moving average window
        
    Returns:
        Smoothed client data with same structure
    """
    smoothed_data = {}
    
    for client_id, rounds_data in client_data.items():
        smoothed_data[client_id] = {}
        for round_num, metrics_data in rounds_data.items():
            smoothed_data[client_id][round_num] = {}
            for metric, values in metrics_data.items():
                if isinstance(values, list) and len(values) > 0:
                    # Filter out None values for smoothing
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        # Apply smoothing
                        series = pd.Series(valid_values)
                        smoothed_series = series.rolling(window=window_size, min_periods=1).mean()
                        
                        # Reconstruct with original structure (including None positions)
                        smoothed_values = []
                        smooth_idx = 0
                        for original_val in values:
                            if original_val is not None:
                                smoothed_values.append(smoothed_series.iloc[smooth_idx])
                                smooth_idx += 1
                            else:
                                smoothed_values.append(None)
                        
                        smoothed_data[client_id][round_num][metric] = smoothed_values
                    else:
                        smoothed_data[client_id][round_num][metric] = values
                else:
                    smoothed_data[client_id][round_num][metric] = values
    
    return smoothed_data

# Federated learning metrics - base metrics without submodel specification
federated_metrics = {
    'Binary_Accuracy/train': 'Training Binary Accuracy',
    'F1/train': 'Training F1 Score',
    'Loss/train': 'Training Loss',
    'Accuracy/val': 'Validation Accuracy',
    'F1/val': 'Validation F1 Score',
    'Loss/val': 'Validation Loss',
    'Binary_Accuracy/test': 'Test Binary Accuracy',
    'F1/test': 'Test F1 Score',
    'Legend': None,
}

line_styles = ['-', '--', '-.', ':']
submodels = {
    'CLAM': line_styles[1],
    'CD': line_styles[2],
    'MM': line_styles[0]
}


def tensorboard_to_datadict_federated(experiment_name: str, fold_num: int, exp_dir: str = ROOT_RESULTS):
    """Extract data from TensorBoard logs for federated learning plotting from a specific fold"""
    
    experiment_dir = os.path.join(exp_dir, experiment_name, f"Fold{fold_num}")
    log_dir = os.path.join(experiment_dir, "log")
    
    if not os.path.exists(log_dir):
        # If not found in the provided directory, try OLD_DATA_ROOT_RESULTS as fallback
        if exp_dir == ROOT_RESULTS:
            old_experiment_dir = os.path.join(OLD_DATA_ROOT_RESULTS, experiment_name, f"Fold{fold_num}")
            old_log_dir = os.path.join(old_experiment_dir, "log")
            if os.path.exists(old_log_dir):
                # print(f"Found experiment {experiment_name} fold {fold_num} in old data directory")
                experiment_dir = old_experiment_dir
                log_dir = old_log_dir
            else:
                print(f"Warning: Log directory {log_dir} does not exist for fold {fold_num} (also checked old data)")
                return {}, {}
        else:
            print(f"Warning: Log directory {log_dir} does not exist for fold {fold_num}")
            return {}, {}
    
    # Separate data structures for clients and server
    client_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    server_data = defaultdict(lambda: defaultdict(dict))
    
    # Parse directory structure
    for item in os.listdir(log_dir):
        if item.startswith("client_") and "round" in item:
            # Parse client training logs: client_{id}_round_{round}
            parts = item.split("_")
            client_id = int(parts[1])
            round_num = int(parts[3])
            
            client_path = os.path.join(log_dir, item)
            if not os.path.exists(client_path):
                continue
                
            for file in os.listdir(client_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(client_path, file)
                    try:
                        for event in tf.compat.v1.train.summary_iterator(event_file):
                            for value in event.summary.value:
                                client_data[client_id][round_num][value.tag].append(value.simple_value)
                    except Exception as e:
                        print(f"Warning: Could not read {event_file}: {e}")
                        continue

        elif "server" in item:
            # Parse server evaluation logs: client_server_{round}
            round_num = int(item.split("_")[2])
            
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
    
    return dict(client_data), dict(server_data)


def get_available_folds(experiment_name: str, exp_dir: str = ROOT_RESULTS):
    """Get list of available fold numbers for an experiment"""
    experiment_dir = os.path.join(exp_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
        # If not found in the provided directory, try OLD_DATA_ROOT_RESULTS as fallback
        if exp_dir == ROOT_RESULTS:
            old_experiment_dir = os.path.join(OLD_DATA_ROOT_RESULTS, experiment_name)
            if os.path.exists(old_experiment_dir):
                print(f"Found experiment {experiment_name} in old data directory")
                experiment_dir = old_experiment_dir
            else:
                raise ValueError(f"Experiment directory {experiment_dir} does not exist (also checked old data)")
        else:
            raise ValueError(f"Experiment directory {experiment_dir} does not exist")
    
    folds = []
    for item in os.listdir(experiment_dir):
        if item.startswith("Fold") and os.path.isdir(os.path.join(experiment_dir, item)):
            try:
                fold_num = int(item[4:])  # Extract number after "Fold"
                folds.append(fold_num)
            except ValueError:
                continue
    
    return sorted(folds)


def create_federated_plot(num_metrics=None):
    """Create figure for federated learning plots"""
    if num_metrics is None:
        num_metrics = len(federated_metrics)
    
    # Calculate layout
    rows_config = {
        1: [1],
        2: [2, 3, 4, 5, 6],
        3: [7, 8, 9, 10, 11, 12],
        4: [13, 14, 15, 16]
    }
    
    num_rows = next(i for i, n in rows_config.items() if num_metrics in n)
    num_cols = (num_metrics + num_rows - 1) // num_rows
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    
    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axs = np.array([axs])
    elif num_rows == 1 or num_cols == 1:
        axs = axs.reshape(num_rows, num_cols)
    
    return fig, axs


def compute_crossfold_experiment_average(experiment_name, fold_numbers, metric, all_rounds, 
                                        global_max_steps_per_round, show_full_training, 
                                        show_individual_clients, smooth_window=0):
    """
    Compute cross-validation average for a single experiment across its folds
    
    Args:
        experiment_name: Name of the experiment
        fold_numbers: List of fold numbers to include
        metric: The metric to analyze
        all_rounds: List of rounds to consider
        global_max_steps_per_round: Maximum steps per round for consistent spacing
        show_full_training: Whether to show full training curves or just final values
        show_individual_clients: Whether to compute stats per client or aggregated
        smooth_window: Window size for smoothing (0 = no smoothing)
        
    Returns:
        Tuple of (client_data, server_data) representing the cross-validation average
    """
    # Load data for all folds
    fold_data = []
    for fold_num in fold_numbers:
        try:
            client_data, server_data = tensorboard_to_datadict_federated(experiment_name, fold_num)
            
            # Apply smoothing if requested
            if smooth_window > 0:
                client_data = smooth_client_data(client_data, smooth_window)
            
            fold_data.append((client_data, server_data))
        except Exception as e:
            print(f"Warning: Could not load data for experiment {experiment_name} fold {fold_num}: {e}")
            continue
    
    if not fold_data:
        return None, None
    
    # Compute server cross-validation average
    server_cv_data = {}
    for round_num in all_rounds:
        round_values = []
        for _, server_data in fold_data:
            if round_num in server_data and metric in server_data[round_num]:
                round_values.append(server_data[round_num][metric])
        
        if round_values:
            server_cv_data[round_num] = np.mean(round_values)
    
    # Compute client cross-validation average
    if show_individual_clients:
        # Compute average per client ID
        client_cv_data = {}
        
        # Get all client IDs across all folds
        all_client_ids = set()
        for client_data, _ in fold_data:
            all_client_ids.update(client_data.keys())
        
        for client_id in sorted(all_client_ids):
            client_cv_data[client_id] = {}
            
            if show_full_training:
                # Average full training curves across folds
                for round_idx, round_num in enumerate(all_rounds):
                    # Collect data for this client/round across all folds
                    fold_curves = []
                    max_steps_this_round = 0
                    
                    for client_data, _ in fold_data:
                        if client_id in client_data and round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if round_values:
                                fold_curves.append(round_values)
                                max_steps_this_round = max(max_steps_this_round, len(round_values))
                    
                    if fold_curves and max_steps_this_round > 0:
                        # Align all folds to same length (pad with last value)
                        aligned_data = []
                        for fold_values in fold_curves:
                            padded_curve = fold_values + [fold_values[-1]] * (max_steps_this_round - len(fold_values))
                            aligned_data.append(padded_curve)
                        
                        # Compute average for each step
                        mean_curve = []
                        for step in range(max_steps_this_round):
                            step_values = [fold_data[step] for fold_data in aligned_data]
                            mean_curve.append(np.mean(step_values))
                        
                        client_cv_data[client_id][round_num] = mean_curve
            
            else:
                # Average final values across folds
                for round_num in all_rounds:
                    final_values = []
                    
                    for client_data, _ in fold_data:
                        if client_id in client_data and round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if round_values:
                                final_values.append(round_values[-1])  # Take final value
                    
                    if final_values:
                        client_cv_data[client_id][round_num] = np.mean(final_values)
    
    else:
        # Compute aggregated average across all clients
        aggregated_cv_data = {}
        
        if show_full_training:
            # Average aggregated full training curves across folds
            for round_idx, round_num in enumerate(all_rounds):
                # Collect aggregated values for this round across all folds
                fold_aggregated = []
                max_steps_this_round = 0
                
                for client_data, _ in fold_data:
                    # Aggregate all clients for this fold/round
                    round_client_data = []
                    
                    for client_id in sorted(client_data.keys()):
                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if round_values:
                                round_client_data.append(round_values)
                                max_steps_this_round = max(max_steps_this_round, len(round_values))
                    
                    if round_client_data:
                        # Average across clients for each step in this round
                        avg_curve = []
                        for step in range(max_steps_this_round):
                            step_values = []
                            for client_curve in round_client_data:
                                if step < len(client_curve):
                                    step_values.append(client_curve[step])
                                else:
                                    step_values.append(client_curve[-1])  # Use last value if shorter
                            
                            if step_values:
                                avg_curve.append(np.mean(step_values))
                        
                        if avg_curve:
                            fold_aggregated.append(avg_curve)
                
                if fold_aggregated and max_steps_this_round > 0:
                    # Align all folds to same length
                    aligned_data = []
                    for fold_curve in fold_aggregated:
                        padded_curve = fold_curve + [fold_curve[-1]] * (max_steps_this_round - len(fold_curve))
                        aligned_data.append(padded_curve)
                    
                    # Compute average for each step
                    mean_curve = []
                    for step in range(max_steps_this_round):
                        step_values = [fold_data[step] for fold_data in aligned_data]
                        mean_curve.append(np.mean(step_values))
                    
                    aggregated_cv_data[round_num] = mean_curve
        
        else:
            # Average aggregated final values across folds
            for round_num in all_rounds:
                fold_aggregated_values = []
                
                for client_data, _ in fold_data:
                    # Collect final values from all clients for this fold/round
                    round_final_values = []
                    
                    for client_id in sorted(client_data.keys()):
                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if round_values:
                                round_final_values.append(round_values[-1])  # Take final value
                    
                    if round_final_values:
                        # Average across clients for this fold
                        fold_aggregated_values.append(np.mean(round_final_values))
                
                if fold_aggregated_values:
                    aggregated_cv_data[round_num] = np.mean(fold_aggregated_values)
        
        client_cv_data = aggregated_cv_data
    
    return client_cv_data, server_cv_data


def compute_crossfold_group_statistics(experiment_group, fold_numbers, metric, all_rounds, 
                                      global_max_steps_per_round, show_full_training, 
                                      show_individual_clients, smooth_window=0):
    """
    Compute statistics across multiple cross-validation experiments in a group
    
    Args:
        experiment_group: List of experiment names in the group
        fold_numbers: List of fold numbers to include for each experiment
        metric: The metric to analyze
        all_rounds: List of rounds to consider
        global_max_steps_per_round: Maximum steps per round for consistent spacing
        show_full_training: Whether to show full training curves or just final values
        show_individual_clients: Whether to compute stats per client or aggregated
        smooth_window: Window size for smoothing (0 = no smoothing)
        
    Returns:
        Tuple of (client_stats, server_stats) where each contains mean and std across experiments
    """
    # Collect cross-validation averages from all experiments in the group
    experiment_cv_averages = []
    
    # print("Computing cross-validation averages for experiments in group...")
    for exp_name in experiment_group:
        client_cv_data, server_cv_data = compute_crossfold_experiment_average(
            exp_name, fold_numbers, metric, all_rounds, global_max_steps_per_round,
            show_full_training, show_individual_clients, smooth_window
        )
        
        if client_cv_data is not None and server_cv_data is not None:
            experiment_cv_averages.append((client_cv_data, server_cv_data))
    
    if not experiment_cv_averages:
        return None, None
    
    # Compute server statistics across experiment cross-validation averages
    server_stats = {}
    for round_num in all_rounds:
        experiment_round_values = []
        
        # Collect cross-validation average for this round from each experiment
        for _, server_cv_data in experiment_cv_averages:
            if round_num in server_cv_data:
                experiment_round_values.append(server_cv_data[round_num])
        
        if experiment_round_values:
            server_stats[round_num] = {
                'mean': np.mean(experiment_round_values),
                'std': np.std(experiment_round_values, ddof=1) if len(experiment_round_values) > 1 else 0.0
            }
    
    # Compute client statistics across experiment cross-validation averages
    if show_individual_clients:
        # Compute statistics per client ID across experiments
        client_stats = {}
        
        # Get all client IDs across all experiments
        all_client_ids = set()
        for client_cv_data, _ in experiment_cv_averages:
            all_client_ids.update(client_cv_data.keys())
        
        for client_id in sorted(all_client_ids):
            client_stats[client_id] = {}
            
            if show_full_training:
                # Statistics for full training curves
                for round_idx, round_num in enumerate(all_rounds):
                    experiment_curves = []
                    max_steps_this_round = 0
                    
                    # Collect cross-validation average curve for this round from each experiment
                    for client_cv_data, _ in experiment_cv_averages:
                        if client_id in client_cv_data and round_num in client_cv_data[client_id]:
                            curve = client_cv_data[client_id][round_num]
                            if isinstance(curve, list) and curve:
                                experiment_curves.append(curve)
                                max_steps_this_round = max(max_steps_this_round, len(curve))
                    
                    if experiment_curves and max_steps_this_round > 0:
                        # Align all experiment curves to same length
                        aligned_exp_data = []
                        for exp_curve in experiment_curves:
                            padded_curve = exp_curve + [exp_curve[-1]] * (max_steps_this_round - len(exp_curve))
                            aligned_exp_data.append(padded_curve)
                        
                        # Compute statistics across experiments for each step
                        mean_curve = []
                        std_curve = []
                        
                        for step in range(max_steps_this_round):
                            step_values = [exp_data[step] for exp_data in aligned_exp_data]
                            mean_curve.append(np.mean(step_values))
                            std_curve.append(np.std(step_values, ddof=1) if len(step_values) > 1 else 0.0)
                        
                        client_stats[client_id][round_num] = {
                            'mean': mean_curve,
                            'std': std_curve
                        }
            
            else:
                # Statistics for final values only
                for round_num in all_rounds:
                    experiment_final_values = []
                    
                    # Collect cross-validation average final value for this round from each experiment
                    for client_cv_data, _ in experiment_cv_averages:
                        if client_id in client_cv_data and round_num in client_cv_data[client_id]:
                            value = client_cv_data[client_id][round_num]
                            if isinstance(value, (int, float)):
                                experiment_final_values.append(value)
                    
                    if experiment_final_values:
                        client_stats[client_id][round_num] = {
                            'mean': np.mean(experiment_final_values),
                            'std': np.std(experiment_final_values, ddof=1) if len(experiment_final_values) > 1 else 0.0
                        }
    
    else:
        # Compute aggregated statistics across all clients and experiments
        aggregated_stats = {}
        
        if show_full_training:
            # Statistics for aggregated full training curves
            for round_idx, round_num in enumerate(all_rounds):
                experiment_curves = []
                max_steps_this_round = 0
                
                # Collect aggregated cross-validation average curve for this round from each experiment
                for client_cv_data, _ in experiment_cv_averages:
                    if round_num in client_cv_data:
                        curve = client_cv_data[round_num]
                        if isinstance(curve, list) and curve:
                            experiment_curves.append(curve)
                            max_steps_this_round = max(max_steps_this_round, len(curve))
                
                if experiment_curves and max_steps_this_round > 0:
                    # Align all experiment curves to same length
                    aligned_exp_data = []
                    for exp_curve in experiment_curves:
                        padded_curve = exp_curve + [exp_curve[-1]] * (max_steps_this_round - len(exp_curve))
                        aligned_exp_data.append(padded_curve)
                    
                    # Compute statistics across experiments for each step
                    mean_curve = []
                    std_curve = []
                    
                    for step in range(max_steps_this_round):
                        step_values = [exp_data[step] for exp_data in aligned_exp_data]
                        mean_curve.append(np.mean(step_values))
                        std_curve.append(np.std(step_values, ddof=1) if len(step_values) > 1 else 0.0)
                    
                    aggregated_stats[round_num] = {
                        'mean': mean_curve,
                        'std': std_curve
                    }
        
        else:
            # Statistics for aggregated final values
            for round_num in all_rounds:
                experiment_aggregated_values = []
                
                # Collect aggregated cross-validation average final value for this round from each experiment
                for client_cv_data, _ in experiment_cv_averages:
                    if round_num in client_cv_data:
                        value = client_cv_data[round_num]
                        if isinstance(value, (int, float)):
                            experiment_aggregated_values.append(value)
                
                if experiment_aggregated_values:
                    aggregated_stats[round_num] = {
                        'mean': np.mean(experiment_aggregated_values),
                        'std': np.std(experiment_aggregated_values, ddof=1) if len(experiment_aggregated_values) > 1 else 0.0
                    }
        
        client_stats = aggregated_stats
    
    return client_stats, server_stats


def plot_crossfold_group_metric(ax, metric, title, client_stats, server_stats, all_rounds, 
                               global_max_steps_per_round, show_full_training, show_individual_clients, 
                               show_std=True, show_legend=True, color='blue', label_prefix=''):
    """
    Plot a single metric for cross-fold group comparison results
    """
    
    # Alpha (transparency) settings
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.3
    
    # For train/val metrics, exclude round 0 (initial evaluation before training)
    if 'test' not in metric:
        plot_rounds = [r for r in all_rounds if r > 0]
    else:
        plot_rounds = all_rounds
    
    if 'test' in metric:
        # For test metrics, plot server evaluation points only
        if server_stats:
            server_means = []
            server_stds = []
            server_rounds = []
            
            for round_num in sorted(server_stats.keys()):
                if round_num in plot_rounds:
                    server_means.append(server_stats[round_num]['mean'])
                    server_stds.append(server_stats[round_num]['std'])
                    server_rounds.append(round_num)
            
            if server_means:
                # Plot mean line
                label = f'{label_prefix} (Group avg)' if label_prefix else 'Group Average'
                ax.plot(server_rounds, server_means, color=color, marker='o', 
                       linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA)
                
                # Plot standard deviation band
                if show_std and any(std > 0 for std in server_stds):
                    server_means = np.array(server_means)
                    server_stds = np.array(server_stds)
                    ax.fill_between(server_rounds, 
                                   server_means - server_stds, 
                                   server_means + server_stds,
                                   color=color, alpha=STD_ALPHA)
                
                ax.set_xlabel('Federated Round')
    
    else:
        # For train/val metrics, plot client training curves
        if not plot_rounds or not client_stats:
            return
            
        if show_individual_clients:
            # Plot individual client statistics with different line styles
            client_line_styles = ['-', '--', '-.', ':']
            
            for client_idx, client_id in enumerate(sorted(client_stats.keys())):
                client_line_style = client_line_styles[client_idx % len(client_line_styles)]
                
                if show_full_training:
                    # Plot full training curves
                    all_means = []
                    all_stds = []
                    all_steps = []
                    
                    max_steps_per_round = global_max_steps_per_round
                    
                    for round_idx, round_num in enumerate(plot_rounds):
                        if round_num in client_stats[client_id]:
                            round_means = client_stats[client_id][round_num]['mean']
                            round_stds = client_stats[client_id][round_num]['std']
                            
                            # Create x-positions: each round starts at round_idx * max_steps_per_round
                            round_start = round_idx * max_steps_per_round
                            round_steps = list(range(round_start, round_start + len(round_means)))
                            
                            all_means.extend(round_means)
                            all_stds.extend(round_stds)
                            all_steps.extend(round_steps)
                    
                    if all_means:
                        # Plot mean line
                        label = f'{label_prefix} C{client_id} (Group avg)' if label_prefix else f'Client {client_id} (Group avg)'
                        ax.plot(all_steps, all_means, color=color, linewidth=2, linestyle=client_line_style,
                               label=label, alpha=LINE_ALPHA)
                        
                        # Plot standard deviation band
                        if show_std and any(std > 0 for std in all_stds):
                            all_means = np.array(all_means)
                            all_stds = np.array(all_stds)
                            ax.fill_between(all_steps, 
                                           all_means - all_stds, 
                                           all_means + all_stds,
                                           color=color, alpha=STD_ALPHA)
                
                else:
                    # Plot final values only
                    round_means = []
                    round_stds = []
                    round_positions = []
                    
                    for round_num in plot_rounds:
                        if round_num in client_stats[client_id]:
                            round_means.append(client_stats[client_id][round_num]['mean'])
                            round_stds.append(client_stats[client_id][round_num]['std'])
                            round_positions.append(round_num)
                    
                    if round_means:
                        # Plot mean line
                        label = f'{label_prefix} C{client_id} (Group avg)' if label_prefix else f'Client {client_id} (Group avg)'
                        ax.plot(round_positions, round_means, color=color, marker='o', linestyle=client_line_style,
                               linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA)
                        
                        # Plot standard deviation error bars
                        if show_std and any(std > 0 for std in round_stds):
                            ax.errorbar(round_positions, round_means, yerr=round_stds,
                                       color=color, alpha=STD_ALPHA, capsize=5)
        
        else:
            # Plot aggregated statistics across all clients
            if show_full_training:
                # Plot aggregated full training curves
                all_means = []
                all_stds = []
                all_steps = []
                
                max_steps_per_round = global_max_steps_per_round
                
                for round_idx, round_num in enumerate(plot_rounds):
                    if round_num in client_stats:
                        round_means = client_stats[round_num]['mean']
                        round_stds = client_stats[round_num]['std']
                        
                        # Create x-positions: each round starts at round_idx * max_steps_per_round
                        round_start = round_idx * max_steps_per_round
                        round_steps = list(range(round_start, round_start + len(round_means)))
                        
                        all_means.extend(round_means)
                        all_stds.extend(round_stds)
                        all_steps.extend(round_steps)
                
                if all_means:
                    # Plot mean line
                    label = f'{label_prefix} (Group avg)' if label_prefix else 'Group Average'
                    ax.plot(all_steps, all_means, color=color, linewidth=2, 
                           label=label, alpha=LINE_ALPHA)
                    
                    # Plot standard deviation band
                    if show_std and any(std > 0 for std in all_stds):
                        all_means = np.array(all_means)
                        all_stds = np.array(all_stds)
                        ax.fill_between(all_steps, 
                                       all_means - all_stds, 
                                       all_means + all_stds,
                                       color=color, alpha=STD_ALPHA)
                
                # Add round boundaries
                if len(plot_rounds) > 1:
                    for round_idx in range(1, len(plot_rounds)):
                        boundary = round_idx * max_steps_per_round
                        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                
                # Set x-axis labels
                if plot_rounds and max_steps_per_round > 0:
                    round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(plot_rounds))]
                    ax.set_xticks(round_centers)
                    ax.set_xticklabels([f'Round {r}' for r in plot_rounds])
            
            else:
                # Plot aggregated final values
                round_means = []
                round_stds = []
                round_positions = []
                
                for round_num in plot_rounds:
                    if round_num in client_stats:
                        round_means.append(client_stats[round_num]['mean'])
                        round_stds.append(client_stats[round_num]['std'])
                        round_positions.append(round_num)
                
                if round_means:
                    # Plot mean line
                    label = f'{label_prefix} (Group avg)' if label_prefix else 'Group Average'
                    ax.plot(round_positions, round_means, color=color, marker='o',
                           linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA)
                    
                    # Plot standard deviation error bars
                    if show_std and any(std > 0 for std in round_stds):
                        ax.errorbar(round_positions, round_means, yerr=round_stds,
                                   color=color, alpha=STD_ALPHA, capsize=5)
    
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def create_legend_subplot(axs, ax_dims, ax_width, metrics_to_plot):
    """Create a dedicated legend subplot"""
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        if base_metric == 'Legend':
            # Get the subplot for legend
            if len(ax_dims) > 1:
                legend_ax = axs[i // ax_width, i % ax_width]
            else:
                legend_ax = axs[i] if hasattr(axs, '__len__') else axs
            
            legend_ax.axis('off')
            # Create simple legend handles
            legend_handles = [
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Group Average (CV)')
            ]
            legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
            legend_ax.set_title('Cross-Fold Group Comparison')
            break


def create_comparison_legend_subplot(axs, ax_dims, ax_width, legend_handles, metrics_to_plot):
    """Create a dedicated legend subplot for comparison"""
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        if base_metric == 'Legend':
            # Get the subplot for legend
            if len(ax_dims) > 1:
                legend_ax = axs[i // ax_width, i % ax_width]
            else:
                legend_ax = axs[i] if hasattr(axs, '__len__') else axs
            
            legend_ax.axis('off')
            legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
            legend_ax.set_title('Cross-Fold Group Comparison')
            break


def plot_crossfold_group_comparison(experiment_groups: dict, submodel: str = 'MM', metric_filter: str = 'test',
                                   show_individual_clients: bool = False, show_full_training: bool = False,
                                   smooth_window: int = 0, show_std: bool = True, folds: list = None):
    """
    Compare groups of cross-fold validation experiments from multiple federated learning experiments
    
    Args:
        experiment_groups: Dict mapping group names to lists of experiment names
                          e.g. {'No Phases': ['no_phases_s1', 'no_phases_s2'], 'Early Stop': ['es_2_8_s1', 'es_2_8_s2']}
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD')
        metric_filter: Which type of metrics to plot ('train', 'val', 'test', or 'all')
        show_individual_clients: If True, show individual client statistics instead of aggregated
        show_full_training: If True, show complete training progress within rounds
        smooth_window: If > 0, apply moving average smoothing with this window size
        show_std: If True, show standard deviation as bands/error bars
        folds: List of specific fold numbers to include (None = use all available)
    """
    
    # Colors for different groups
    group_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Validate and prepare experiment groups
    all_group_data = {}
    for group_name, experiment_list in experiment_groups.items():
        if folds is None:
            # Use the first experiment to determine available folds
            available_folds = get_available_folds(experiment_list[0])
            if not available_folds:
                print(f"Warning: No folds found for experiments in group {group_name}")
                continue
            fold_numbers = available_folds
        else:
            fold_numbers = folds
        
        # Validate that all experiments in the group have the required folds
        valid_experiments = []
        for exp_name in experiment_list:
            exp_folds = get_available_folds(exp_name)
            if all(fold in exp_folds for fold in fold_numbers):
                valid_experiments.append(exp_name)
            else:
                print(f"Warning: Experiment {exp_name} missing some required folds")
        
        if valid_experiments:
            # print(f"Using folds for group {group_name}: {fold_numbers}")
            # print(f"  Valid experiments in group: {valid_experiments}")
            all_group_data[group_name] = {
                'experiments': valid_experiments,
                'folds': fold_numbers
            }
        else:
            print(f"Warning: No valid experiments found for group {group_name}")
    
    if not all_group_data:
        raise ValueError("No valid experiment groups found")
    
    # Filter metrics based on metric_filter
    if metric_filter == 'all':
        metrics_to_plot = {k: v for k, v in federated_metrics.items() if k is not None and (v is not None or k == 'Legend')}
    else:
        metrics_to_plot = {k: v for k, v in federated_metrics.items() 
                          if k is not None and (v is not None or k == 'Legend') and (metric_filter in k or k == 'Legend')}
    
    # Create plot
    fig, axs = create_federated_plot(len(metrics_to_plot))
    
    mode_desc = []
    if show_individual_clients:
        mode_desc.append("individual clients")
    else:
        mode_desc.append("aggregated clients")
    if show_full_training:
        mode_desc.append("full training curves")
    else:
        mode_desc.append("final values")
    if smooth_window > 0:
        mode_desc.append(f"smoothed (w={smooth_window})")
    if show_std:
        mode_desc.append("with std dev")
    
    title_suffix = ", ".join(mode_desc)
    group_names_str = " vs ".join(all_group_data.keys())
    fig.suptitle(f"Cross-Fold Group Comparison: {group_names_str}\n({submodel} - {metric_filter} metrics, {title_suffix})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Calculate global parameters from all groups and experiments
    global_all_rounds = set()
    global_max_steps_per_round = 0
    
    # print("Calculating global rounds and max steps per round across all groups...")
    for group_name, group_data in all_group_data.items():
        for exp_name in group_data['experiments']:
            for fold_num in group_data['folds']:
                try:
                    client_data, server_data = tensorboard_to_datadict_federated(exp_name, fold_num)
                    for client_id in client_data.keys():
                        global_all_rounds.update(client_data[client_id].keys())
                        for round_num in client_data[client_id].keys():
                            for metric_key in client_data[client_id][round_num].keys():
                                if client_data[client_id][round_num][metric_key]:
                                    steps_in_round = len([v for v in client_data[client_id][round_num][metric_key] if v is not None])
                                    global_max_steps_per_round = max(global_max_steps_per_round, steps_in_round)
                    global_all_rounds.update(server_data.keys())
                except Exception as e:
                    print(f"Warning: Could not load data for {exp_name} fold {fold_num}: {e}")
                    continue
    
    global_all_rounds = sorted(global_all_rounds)
    
    # Compute statistics for all groups
    all_group_stats = {}
    for group_name, group_data in all_group_data.items():
        print(f"Computing cross-fold group statistics for: {group_name}")
        all_group_stats[group_name] = {}
        
        for base_metric in metrics_to_plot.keys():
            if base_metric == 'Legend':
                continue
            metric = f"{base_metric}/{submodel}"
            
            client_stats, server_stats = compute_crossfold_group_statistics(
                group_data['experiments'], group_data['folds'], metric, global_all_rounds, 
                global_max_steps_per_round, show_full_training, show_individual_clients, smooth_window
            )
            
            all_group_stats[group_name][metric] = (client_stats, server_stats)
    
    ax_dims = axs.shape if hasattr(axs, 'shape') else (1, 1)
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    has_legend_entry = 'Legend' in federated_metrics
    
    # Plot each metric
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        if base_metric == 'Legend':
            continue
            
        # Get the subplot
        if len(ax_dims) > 1:
            ax = axs[i // ax_width, i % ax_width]
        else:
            ax = axs[i] if hasattr(axs, '__len__') else axs
        
        metric = f"{base_metric}/{submodel}"
        
        # Plot each group's results
        for group_idx, (group_name, group_data) in enumerate(all_group_data.items()):
            if group_name not in all_group_stats or metric not in all_group_stats[group_name]:
                continue
                
            client_stats, server_stats = all_group_stats[group_name][metric]
            color = group_colors[group_idx % len(group_colors)]
            
            plot_crossfold_group_metric(ax, metric, title, client_stats, server_stats, global_all_rounds,
                                       global_max_steps_per_round, show_full_training, show_individual_clients,
                                       show_std, show_legend=not has_legend_entry, color=color, 
                                       label_prefix=group_name)
    
    # Create comparison legend if needed
    if has_legend_entry:
        comparison_legend_handles = []
        for group_idx, group_name in enumerate(all_group_data.keys()):
            color = group_colors[group_idx % len(group_colors)]
            line = plt.Line2D([0], [0], color=color, linewidth=2, label=f'{group_name} (Group avg)')
            comparison_legend_handles.append(line)
        
        create_comparison_legend_subplot(axs, ax_dims, ax_width, comparison_legend_handles, metrics_to_plot)
    
    # Hide empty subplots
    total_plots = len(metrics_to_plot)
    if hasattr(axs, 'flat'):
        for j in range(total_plots, len(axs.flat)):
            axs.flat[j].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    suffix_parts = []
    if show_individual_clients:
        suffix_parts.append("individual")
    else:
        suffix_parts.append("aggregated")
    
    if show_full_training:
        suffix_parts.append("fulltraining")
    else:
        suffix_parts.append("finalvals")
    
    if smooth_window > 0:
        suffix_parts.append(f"smooth{smooth_window}")
        
    if show_std:
        suffix_parts.append("withstd")
    
    suffix = "_".join(suffix_parts)
    group_names_clean = "_vs_".join([name.replace("_", "-") for name in all_group_data.keys()])
    plot_path = os.path.join(ROOT_RESULTS, f"crossfold_group_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Cross-fold group comparison plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Fold Group Comparison Plotting Script for Federated Learning')
    parser.add_argument('--groups', type=str, required=True,
                       help='JSON string defining experiment groups, e.g., \'{"No Phases": ["no_phases_s1", "no_phases_s2"], "Early Stop": ["es_2_8_s1", "es_2_8_s2"]}\'')
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD'], 
                       help='Which submodel to plot')
    parser.add_argument('--metric_filter', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                       help='Which metrics to show')
    parser.add_argument('--show_individual_clients', action='store_true', 
                       help='Show individual client statistics instead of aggregated')
    parser.add_argument('--show_full_training', action='store_true',
                       help='Show complete training progress within rounds instead of just final values')
    parser.add_argument('--smooth_window', type=int, default=0,
                       help='Apply moving average smoothing with this window size (0 = no smoothing)')
    parser.add_argument('--show_std', action='store_true', default=True,
                       help='Show standard deviation as bands/error bars')
    parser.add_argument('--no_std', action='store_true',
                       help='Do not show standard deviation (overrides --show_std)')
    parser.add_argument('--folds', nargs='+', type=int,
                       help='Specific fold numbers to include (default: use all available)')
    
    args = parser.parse_args()
    
    # Parse experiment groups from JSON
    try:
        experiment_groups = json.loads(args.groups)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for experiment groups")
        sys.exit(1)
    
    # Handle std flag logic
    show_std = args.show_std and not args.no_std
    
    plot_crossfold_group_comparison(experiment_groups, args.submodel, args.metric_filter,
                                   args.show_individual_clients, args.show_full_training,
                                   args.smooth_window, show_std, args.folds)

# Example usage:

# Compare groups of cross-fold validation experiments (each experiment has multiple folds):
# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["no_phases_s1", "no_phases_s2", "no_phases_s3"], "Early Stop": ["es_2_8_s1", "es_2_8_s2", "es_2_8_s3"]}' --submodel MM --metric_filter test --show_std

# Compare multiple overfitting prevention strategies:
# python plot_crossfold_group_comparison.py --groups '{"Conservative": ["es_2_8_s1", "es_2_8_s2", "es_2_8_s3"], "Medium": ["es_3_10_s1", "es_3_10_s2", "es_3_10_s3"], "Aggressive": ["es_1_6_s1", "es_1_6_s2", "es_1_6_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# Compare with all metrics and specific folds:
# python plot_crossfold_group_comparison.py --groups '{"Group1": ["exp1", "exp2"], "Group2": ["exp3", "exp4"]}' --submodel MM --metric_filter all --folds 0 1 2 3 4 --show_std

# Based on your experiments from the sweep:
# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["no_phases_s1", "no_phases_s2", "no_phases_s3"], "Conservative ES": ["es_2_8_s1", "es_2_8_s2", "es_2_8_s3"], "Medium ES": ["es_3_10_s1", "es_3_10_s2", "es_3_10_s3"], "Aggressive ES": ["es_1_6_s1", "es_1_6_s2", "es_1_6_s3"]}' --submodel MM --metric_filter all --folds 0 1 2 3 4 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases Split 1": ["no_phases_s1", "no_phases_s2", "no_phases_s3"], "No Phases Split 2": ["no_phases_sp2_s1", "no_phases_sp2_s2", "no_phases_sp2_s3"], "No Phases Split 3": ["no_phases_sp3_s1", "no_phases_sp3_s2", "no_phases_sp3_s3"], "No Phases Split 4": ["no_phases_sp4_s1", "no_phases_sp4_s2", "no_phases_sp4_s3"], "No Phases Split 5": ["no_phases_sp5_s1", "no_phases_sp5_s2", "no_phases_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases Split 1": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "No Phases Split 2": ["nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3"], "No Phases Split 3": ["nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3"], "No Phases Split 4": ["nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3"], "No Phases Split 5": ["nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"],"Augmented Split 1": ["aug_no_phases_s1", "aug_no_phases_s2", "aug_no_phases_s3"], "Augmented Split 2": ["aug_no_phases_sp2_s1", "aug_no_phases_sp2_s2", "aug_no_phases_sp2_s3"], "Augmented Split 3": ["aug_no_phases_sp3_s1", "aug_no_phases_sp3_s2", "aug_no_phases_sp3_s3"], "Augmented Split 4": ["aug_no_phases_sp4_s1", "aug_no_phases_sp4_s2", "aug_no_phases_sp4_s3"], "Augmented Split 5": ["aug_no_phases_sp5_s1", "aug_no_phases_sp5_s2", "aug_no_phases_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases Split 1": ["folds_sp1_4115_s1", "folds_sp1_4115_s2", "folds_sp1_4115_s3", "folds_sp1_4115_s4", "folds_sp1_4115_s5"], "No Phases Split 2": ["folds_sp2_4115_s1", "folds_sp2_4115_s2", "folds_sp2_4115_s3", "folds_sp2_4115_s4", "folds_sp2_4115_s5"], "No Phases Split 3": ["folds_sp3_4115_s1", "folds_sp3_4115_s2", "folds_sp3_4115_s3", "folds_sp3_4115_s4", "nfolds_sp3_4115_s5"], "No Phases Split 4": ["folds_sp4_4115_s1", "folds_sp4_4115_s2", "folds_sp4_4115_s3", "folds_sp4_4115_s4", "folds_sp4_4115_s5"], "No Phases Split 5": ["folds_sp5_4115_s1", "folds_sp5_4115_s2", "folds_sp5_4115_s3", "folds_sp5_4115_s4", "folds_sp5_4115_s5"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["no_phases_s1", "no_phases_s2", "no_phases_s3", "no_phases_sp2_s1", "no_phases_sp2_s2", "no_phases_sp2_s3", "no_phases_sp3_s1", "no_phases_sp3_s2", "no_phases_sp3_s3", "no_phases_sp4_s1", "no_phases_sp4_s2", "no_phases_sp4_s3", "no_phases_sp5_s1", "no_phases_sp5_s2", "no_phases_sp5_s3"], "No Phases New": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3", "nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3", "nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3", "nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3", "nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"],"5_4_11": ["folds_sp1_4115_s1", "folds_sp1_4115_s2", "folds_sp1_4115_s3", "folds_sp2_4115_s1", "folds_sp2_4115_s2", "folds_sp2_4115_s3", "folds_sp3_4115_s1", "folds_sp3_4115_s2", "folds_sp3_4115_s3", "folds_sp4_4115_s1", "folds_sp4_4115_s2", "folds_sp4_4115_s3", "folds_sp5_4115_s1", "folds_sp5_4115_s2", "folds_sp5_4115_s3"],"5_4_11 New": ["nda_folds_4115_s1", "nda_folds_4115_s2", "nda_folds_4115_s3", "nda_folds_4115_sp2_s1", "nda_folds_4115_sp2_s2", "nda_folds_4115_sp2_s3", "nda_folds_4115_sp3_s1", "nda_folds_4115_sp3_s2", "nda_folds_4115_sp3_s3", "nda_folds_4115_sp4_s1", "nda_folds_4115_sp4_s2", "nda_folds_4115_sp4_s3", "nda_folds_4115_sp5_s1", "nda_folds_4115_sp5_s2", "nda_folds_4115_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3", "nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3", "nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3", "nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3", "nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"], "Augmented": ["aug_no_phases_s1", "aug_no_phases_s2", "aug_no_phases_s3", "aug_no_phases_sp2_s1", "aug_no_phases_sp2_s2", "aug_no_phases_sp2_s3", "aug_no_phases_sp3_s1", "aug_no_phases_sp3_s2", "aug_no_phases_sp3_s3", "aug_no_phases_sp4_s1", "aug_no_phases_sp4_s2", "aug_no_phases_sp4_s3", "aug_no_phases_sp5_s1", "aug_no_phases_sp5_s2", "aug_no_phases_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "nda_no_phases_5ep": ["nda_no_phases_5ep_s1", "nda_no_phases_5ep_s2", "nda_no_phases_5ep_s3"], "nda_no_phases_5ep_es31": ["nda_no_phases_5ep_es31_s1", "nda_no_phases_5ep_es31_s2", "nda_no_phases_5ep_es31_s3"], "nda_no_phases_10ep_es21": ["nda_no_phases_10ep_es21_s1", "nda_no_phases_10ep_es21_s2", "nda_no_phases_10ep_es21_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std


# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["no_phases_s1", "no_phases_s2", "no_phases_s3"], "New Data No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "ND 4115": ["nda_folds_4115_s1","nda_folds_4115_s2", "nda_folds_4115_s3"], "No Phases but first": ["nda_no_phases_but_first_sp1_s1", "nda_no_phases_but_first_sp1_s2", "nda_no_phases_but_first_sp1_s3"], "Augmented": ["aug_no_phases_s1", "aug_no_phases_s2", "aug_no_phases_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --show_individual_clients

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["no_phases_s1", "no_phases_s2", "no_phases_s3"], "New Data No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "Augmented": ["aug_no_phases_s1", "aug_no_phases_s2", "aug_no_phases_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --show_individual_clients

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3", "nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3", "nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3", "nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3", "nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases Seed 1": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "Random Seed 1": ["Random_s1", "Random_s2", "Random_s3"],"No Phases  Seed 2": ["nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3"], "Random Seed 2": ["Random_sp2_s1", "Random_sp2_s2", "Random_sp2_s3"],"No Phases  Seed 3": ["nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3"], "Random Seed 3": ["Random_sp3_s1", "Random_sp3_s2", "Random_sp3_s3"],"No Phases  Seed 4": ["nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3"], "Random Seed 4": ["Random_sp4_s1", "Random_sp4_s2", "Random_sp4_s3"],"No Phases Seed 5": ["nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"], "Random Seed 5": ["Random_sp5_s1", "Random_sp5_s2", "Random_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases Split 1": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "Centralized Split 1": ["nda_no_phases_centralized_s1", "nda_no_phases_centralized_s2", "nda_no_phases_centralized_s3"],"No Phases Split 2": ["nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3"], "Centralized Split 2": ["nda_no_phases_centralized_sp2_s1", "nda_no_phases_centralized_sp2_s2", "nda_no_phases_centralized_sp2_s3"],"No Phases Split 3": ["nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3"], "Centralized Split 3": ["nda_no_phases_centralized_sp3_s1", "nda_no_phases_centralized_sp3_s2", "nda_no_phases_centralized_sp3_s3"],"No Phases Split 4": ["nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3"], "Centralized Split 4": ["nda_no_phases_centralized_sp4_s1", "nda_no_phases_centralized_sp4_s2", "nda_no_phases_centralized_sp4_s3"],"No Phases Split 5": ["nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"], "Centralized Split 5": ["nda_no_phases_centralized_sp5_s1", "nda_no_phases_centralized_sp5_s2", "nda_no_phases_centralized_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3", "nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3", "nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3", "nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3", "nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"],"Centralized": ["nda_no_phases_centralized_s1", "nda_no_phases_centralized_s2", "nda_no_phases_centralized_s3", "nda_no_phases_centralized_sp2_s1", "nda_no_phases_centralized_sp2_s2", "nda_no_phases_centralized_sp2_s3", "nda_no_phases_centralized_sp3_s1", "nda_no_phases_centralized_sp3_s2", "nda_no_phases_centralized_sp3_s3", "nda_no_phases_centralized_sp4_s1", "nda_no_phases_centralized_sp4_s2", "nda_no_phases_centralized_sp4_s3", "nda_no_phases_centralized_sp5_s1", "nda_no_phases_centralized_sp5_s2", "nda_no_phases_centralized_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std


# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3", "nda_no_phases_sp2_s1", "nda_no_phases_sp2_s2", "nda_no_phases_sp2_s3", "nda_no_phases_sp3_s1", "nda_no_phases_sp3_s2", "nda_no_phases_sp3_s3", "nda_no_phases_sp4_s1", "nda_no_phases_sp4_s2", "nda_no_phases_sp4_s3", "nda_no_phases_sp5_s1", "nda_no_phases_sp5_s2", "nda_no_phases_sp5_s3"],"Random": ["Random_s1","Random_s2","Random_s3", "Random_sp2_s1", "Random_sp2_s2", "Random_sp2_s3", "Random_sp3_s1","Random_sp3_s2","Random_sp3_s3", "Random_sp4_s1","Random_sp4_s2","Random_sp4_s3", "Random_sp5_s1","Random_sp5_s2","Random_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"No Phases": ["nda_no_phases_s1", "nda_no_phases_s2", "nda_no_phases_s3"], "No Phases Long": ["nda_no_phases_3ep_s1", "nda_no_phases_3ep_s2", "nda_no_phases_3ep_s3"], "No Phases 2 Epochs": ["nda_no_phases_2ep_s1", "nda_no_phases_2ep_s2", "nda_no_phases_2ep_s3"], "2:10, 5:3, 2": ["nda_no_phases_epsteps2:10_5:3_2_s1", "nda_no_phases_epsteps2:10_5:3_2_s2", "nda_no_phases_epsteps2:10_5:3_2_s3"], "1:10, 2:5, 3": ["nda_no_phases_epsteps1:10_2:5_3_s1", "nda_no_phases_epsteps1:10_2:5_3_s2", "nda_no_phases_epsteps1:10_2:5_3_s3"], "No Phases 1 Epoch": ["nda_no_phases_1ep_s1", "nda_no_phases_1ep_s2", "nda_no_phases_1ep_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --show_individual_clients