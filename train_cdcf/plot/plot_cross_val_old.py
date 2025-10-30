import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cdcf/results"

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
# Test metrics now support two stages:
# - stage0: Initial test evaluation (before training, available from round 0)
# - stageR: Post-training test evaluation (after training, available from middle rounds)
federated_metrics = {
    'Binary_Accuracy/train': 'Training Binary Accuracy',
    'F1/train': 'Training F1 Score',
    'Loss/train': 'Training Loss',
    'Accuracy/val': 'Validation Accuracy',
    'F1/val': 'Validation F1 Score',
    'Loss/val': 'Validation Loss',
    'Binary_Accuracy/test/stage0': 'Test Binary Accuracy (Initial)',
    'F1/test/stage0': 'Test F1 Score (Initial)',
    'Binary_Accuracy/test/stageR': 'Test Binary Accuracy (Post-Training)',
    'F1/test/stageR': 'Test F1 Score (Post-Training)',
    'Legend': None,
}

line_styles = ['-', '--', '-.', ':']
submodels = {
    'CLAM': line_styles[1],
    'CD': line_styles[2],
    'MM': line_styles[0]
}

def construct_metric_name(base_metric, submodel, stage=None):
    """
    Construct the full metric name for TensorBoard logs.
    
    Args:
        base_metric: Base metric like 'F1/train' or 'Binary_Accuracy/test/stage0'
        submodel: Submodel name ('MM', 'CLAM', 'CD')
        stage: Stage number for test metrics (None for train/val metrics)
    
    Returns:
        Full metric name for TensorBoard lookup
    """
    if 'test' in base_metric:
        if '/stage0' in base_metric:
            # Stage 0: Initial evaluation (stage = 0)
            base_without_stage = base_metric.replace('/stage0', '')
            return f"{base_without_stage}/{submodel}/0"
        elif '/stageR' in base_metric:
            # Stage R: Post-training evaluation (stage = round_num, will be handled dynamically)
            base_without_stage = base_metric.replace('/stageR', '')
            return f"{base_without_stage}/{submodel}"  # Will need round-specific lookup
        else:
            # Legacy format
            return f"{base_metric}/{submodel}"
    else:
        return f"{base_metric}/{submodel}"


def tensorboard_to_datadict_federated(experiment_name: str, fold_num: int, exp_dir: str = ROOT_RESULTS):
    """
    Extract data from TensorBoard logs for federated learning plotting from a specific fold.
    
    Note: Handles both old format (F1/test/MM) and new format (F1/test/MM/stage) test metrics.
    Creates backward compatibility entries for new format metrics.
    """
    
    experiment_dir = os.path.join(exp_dir, experiment_name, f"Fold{fold_num}")
    log_dir = os.path.join(experiment_dir, "log")
    
    if not os.path.exists(log_dir):
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
                                # Store the original metric
                                client_data[client_id][round_num][value.tag].append(value.simple_value)
                                
                                # Create stage-specific entries for test metrics
                                # New format: 'F1/test/MM/0' -> 'F1/test/MM/stage0' and 'F1/test/MM/stageR'
                                if '/test/' in value.tag and value.tag.count('/') >= 3:
                                    parts = value.tag.split('/')
                                    if len(parts) >= 4:  # metric/test/submodel/stage
                                        metric_name = parts[0]
                                        test_part = parts[1]
                                        submodel_part = parts[2]
                                        stage_part = parts[3]
                                        
                                        # Create stage-specific tags
                                        if stage_part == '0':
                                            # Stage 0: Initial evaluation
                                            stage0_tag = f"{metric_name}/{test_part}/stage0/{submodel_part}"
                                            client_data[client_id][round_num][stage0_tag].append(value.simple_value)
                                        else:
                                            # Stage R: Post-training evaluation (stage = round_num)
                                            stageR_tag = f"{metric_name}/{test_part}/stageR/{submodel_part}"
                                            client_data[client_id][round_num][stageR_tag].append(value.simple_value)
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
                                # Store the original metric
                                server_data[round_num][value.tag] = value.simple_value
                                
                                # Create stage-specific entries for test metrics
                                # New format: 'F1/test/MM/0' -> 'F1/test/MM/stage0' and 'F1/test/MM/stageR'
                                if '/test/' in value.tag and value.tag.count('/') >= 3:
                                    parts = value.tag.split('/')
                                    if len(parts) >= 4:  # metric/test/submodel/stage
                                        metric_name = parts[0]
                                        test_part = parts[1]
                                        submodel_part = parts[2]
                                        stage_part = parts[3]
                                        
                                        # Create stage-specific tags
                                        if stage_part == '0':
                                            # Stage 0: Initial evaluation
                                            stage0_tag = f"{metric_name}/{test_part}/stage0/{submodel_part}"
                                            server_data[round_num][stage0_tag] = value.simple_value
                                        else:
                                            # Stage R: Post-training evaluation (stage = round_num)
                                            stageR_tag = f"{metric_name}/{test_part}/stageR/{submodel_part}"
                                            server_data[round_num][stageR_tag] = value.simple_value
                    except Exception as e:
                        print(f"Warning: Could not read {event_file}: {e}")
                        continue
    
    return dict(client_data), dict(server_data)


def get_available_folds(experiment_name: str, exp_dir: str = ROOT_RESULTS):
    """Get list of available fold numbers for an experiment"""
    experiment_dir = os.path.join(exp_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
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


def get_metric_value(data_dict, metric, round_num):
    """
    Get metric value from data dictionary, handling dynamic stage lookup for stageR metrics.
    
    Args:
        data_dict: Dictionary with metric data
        metric: Metric name (may contain stageR)
        round_num: Current round number
        
    Returns:
        Metric value if found, None otherwise
    """
    if '/stageR/' in metric:
        # For stageR metrics, we need to look for the actual stage = round_num
        base_metric = metric.replace('/stageR/', '/test/')
        # Try to find the metric with stage = round_num
        stage_metric = f"{base_metric}/{round_num}"
        if stage_metric in data_dict:
            return data_dict[stage_metric]
        # If not found, return None (this stage may not exist for this round)
        return None
    else:
        # Regular metric lookup
        return data_dict.get(metric, None)


def compute_fold_statistics(experiment_name, fold_numbers, metric, all_rounds, global_max_steps_per_round, 
                           show_full_training, show_individual_clients, smooth_window=0):
    """
    Compute mean and standard deviation statistics across folds for cross-validation
    
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
        Tuple of (client_stats, server_stats)
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
            print(f"Warning: Could not load data for fold {fold_num}: {e}")
            continue
    
    if not fold_data:
        return None, None
    
    # Compute server statistics (simpler case)
    server_stats = {}
    for round_num in all_rounds:
        round_values = []
        for _, server_data in fold_data:
            if round_num in server_data:
                value = get_metric_value(server_data[round_num], metric, round_num)
                if value is not None:
                    round_values.append(value)
        
        if round_values:
            server_stats[round_num] = {
                'mean': np.mean(round_values),
                'std': np.std(round_values, ddof=1) if len(round_values) > 1 else 0.0
            }
    
    # Compute client statistics
    if show_individual_clients:
        # Compute statistics per client ID
        client_stats = {}
        
        # Get all client IDs across all folds
        all_client_ids = set()
        for client_data, _ in fold_data:
            all_client_ids.update(client_data.keys())
        
        for client_id in sorted(all_client_ids):
            client_stats[client_id] = {}
            
            if show_full_training:
                # Statistics for full training curves
                for round_idx, round_num in enumerate(all_rounds):
                    # Collect data for this client/round across all folds
                    fold_curves = []
                    max_steps_this_round = 0
                    
                    for client_data, _ in fold_data:
                        if client_id in client_data and round_num in client_data[client_id]:
                            round_values = get_metric_value(client_data[client_id][round_num], metric, round_num)
                            if round_values is not None:
                                # round_values could be a list (full training) or single value
                                if isinstance(round_values, list):
                                    round_values = [v for v in round_values if v is not None]
                                    if round_values:
                                        fold_curves.append(round_values)
                                        max_steps_this_round = max(max_steps_this_round, len(round_values))
                                else:
                                    # Single value - treat as one-step curve
                                    fold_curves.append([round_values])
                                    max_steps_this_round = max(max_steps_this_round, 1)
                    
                    if fold_curves and max_steps_this_round > 0:
                        # Align all folds to same length (pad with last value)
                        aligned_data = []
                        for fold_values in fold_curves:
                            padded_curve = fold_values + [fold_values[-1]] * (max_steps_this_round - len(fold_values))
                            aligned_data.append(padded_curve)
                        
                        # Compute statistics for each step
                        mean_curve = []
                        std_curve = []
                        
                        for step in range(max_steps_this_round):
                            step_values = [fold_data[step] for fold_data in aligned_data]
                            mean_curve.append(np.mean(step_values))
                            std_curve.append(np.std(step_values, ddof=1) if len(step_values) > 1 else 0.0)
                        
                        client_stats[client_id][round_num] = {
                            'mean': mean_curve,
                            'std': std_curve
                        }
            
            else:
                # Statistics for final values only
                for round_num in all_rounds:
                    final_values = []
                    
                    for client_data, _ in fold_data:
                        if client_id in client_data and round_num in client_data[client_id]:
                            round_values = get_metric_value(client_data[client_id][round_num], metric, round_num)
                            if round_values is not None:
                                # Handle both list and single value cases
                                if isinstance(round_values, list):
                                    filtered_values = [v for v in round_values if v is not None]
                                    if filtered_values:
                                        final_values.append(filtered_values[-1])  # Take final value
                                else:
                                    final_values.append(round_values)
                    
                    if final_values:
                        client_stats[client_id][round_num] = {
                            'mean': np.mean(final_values),
                            'std': np.std(final_values, ddof=1) if len(final_values) > 1 else 0.0
                        }
    
    else:
        # Compute aggregated statistics across all clients
        aggregated_stats = {}
        
        if show_full_training:
            # Statistics for aggregated full training curves
            for round_idx, round_num in enumerate(all_rounds):
                # Collect aggregated values for this round across all folds
                fold_aggregated = []
                max_steps_this_round = 0
                
                for client_data, _ in fold_data:
                    # Aggregate all clients for this fold/round
                    round_client_data = []
                    
                    for client_id in sorted(client_data.keys()):
                        if round_num in client_data[client_id]:
                            round_values = get_metric_value(client_data[client_id][round_num], metric, round_num)
                            if round_values is not None:
                                # Handle both list and single value cases
                                if isinstance(round_values, list):
                                    filtered_values = [v for v in round_values if v is not None]
                                    if filtered_values:
                                        round_client_data.append(filtered_values)
                                        max_steps_this_round = max(max_steps_this_round, len(filtered_values))
                                else:
                                    # Single value - treat as one-step curve
                                    round_client_data.append([round_values])
                                    max_steps_this_round = max(max_steps_this_round, 1)
                    
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
                    
                    # Compute statistics for each step
                    mean_curve = []
                    std_curve = []
                    
                    for step in range(max_steps_this_round):
                        step_values = [fold_data[step] for fold_data in aligned_data]
                        mean_curve.append(np.mean(step_values))
                        std_curve.append(np.std(step_values, ddof=1) if len(step_values) > 1 else 0.0)
                    
                    aggregated_stats[round_num] = {
                        'mean': mean_curve,
                        'std': std_curve
                    }
        
        else:
            # Statistics for aggregated final values
            for round_num in all_rounds:
                fold_aggregated_values = []
                
                for client_data, _ in fold_data:
                    # Collect final values from all clients for this fold/round
                    round_final_values = []
                    
                    for client_id in sorted(client_data.keys()):
                        if round_num in client_data[client_id]:
                            round_values = get_metric_value(client_data[client_id][round_num], metric, round_num)
                            if round_values is not None:
                                # Handle both list and single value cases
                                if isinstance(round_values, list):
                                    filtered_values = [v for v in round_values if v is not None]
                                    if filtered_values:
                                        round_final_values.append(filtered_values[-1])  # Take final value
                                else:
                                    round_final_values.append(round_values)
                    
                    if round_final_values:
                        # Average across clients for this fold
                        fold_aggregated_values.append(np.mean(round_final_values))
                
                if fold_aggregated_values:
                    aggregated_stats[round_num] = {
                        'mean': np.mean(fold_aggregated_values),
                        'std': np.std(fold_aggregated_values, ddof=1) if len(fold_aggregated_values) > 1 else 0.0
                    }
        
        client_stats = aggregated_stats
    
    return client_stats, server_stats


def plot_crossval_metric(ax, metric, title, client_stats, server_stats, all_rounds, 
                        global_max_steps_per_round, show_full_training, show_individual_clients, 
                        show_std=True, show_legend=True):
    """
    Plot a single metric for cross-validation results
    
    Args:
        ax: Matplotlib axis to plot on
        metric: The metric being plotted
        title: Title for the plot
        client_stats: Client statistics from compute_fold_statistics
        server_stats: Server statistics from compute_fold_statistics
        all_rounds: List of rounds to plot
        global_max_steps_per_round: Maximum steps per round for spacing
        show_full_training: Whether showing full training curves
        show_individual_clients: Whether showing individual clients
        show_std: Whether to show standard deviation bands
        show_legend: Whether to show legend
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
        # For test metrics, show individual clients if requested
        if show_individual_clients and client_stats:
            # Plot individual client test statistics
            client_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for client_idx, client_id in enumerate(sorted(client_stats.keys())):
                color = client_colors[client_idx % len(client_colors)]
                
                # For test metrics, we only have final values (no full training curves)
                client_means = []
                client_stds = []
                client_rounds = []
                
                for round_num in sorted(client_stats[client_id].keys()):
                    if round_num in plot_rounds:
                        mean_val = client_stats[client_id][round_num]['mean']
                        std_val = client_stats[client_id][round_num]['std']
                        
                        # Handle both single values and lists
                        if isinstance(mean_val, list):
                            mean_val = mean_val[-1] if mean_val else 0.0  # Take last value if list
                        if isinstance(std_val, list):
                            std_val = std_val[-1] if std_val else 0.0    # Take last value if list
                            
                        client_means.append(mean_val)
                        client_stds.append(std_val)
                        client_rounds.append(round_num)
                
                if client_means:
                    # Plot client mean line with dashed style
                    ax.plot(client_rounds, client_means, color=color, marker='o', linestyle='--',
                           linewidth=2, markersize=4, label=f'Client {client_id} (CV avg)', alpha=LINE_ALPHA)
                    
                    # Plot standard deviation shaded band for individual clients
                    if show_std and client_stds and any(std > 0 for std in client_stds):
                        client_means_array = np.array(client_means)
                        client_stds_array = np.array(client_stds)
                        ax.fill_between(client_rounds, 
                                       client_means_array - client_stds_array, 
                                       client_means_array + client_stds_array,
                                       color=color, alpha=STD_ALPHA * 0.6)  # Lighter shade than server
        
        # Always show server evaluation for test metrics
        if server_stats:
            # Default behavior: plot server evaluation points only
            server_means = []
            server_stds = []
            server_rounds = []
            
            for round_num in sorted(server_stats.keys()):
                if round_num in plot_rounds:
                    server_means.append(server_stats[round_num]['mean'])
                    server_stds.append(server_stats[round_num]['std'])
                    server_rounds.append(round_num)
            
            if server_means:
                # Plot mean line - use black with square markers to distinguish from clients
                server_color = 'black' if show_individual_clients and client_stats else 'blue'
                server_marker = 's' if show_individual_clients and client_stats else 'o'
                server_label = 'Global Model (CV avg)' if show_individual_clients and client_stats else 'Cross-Val Average'
                ax.plot(server_rounds, server_means, color=server_color, marker=server_marker, 
                       linewidth=2, markersize=6, label=server_label, alpha=LINE_ALPHA)
                
                # Plot standard deviation band
                if show_std and any(std > 0 for std in server_stds):
                    server_means = np.array(server_means)
                    server_stds = np.array(server_stds)
                    ax.fill_between(server_rounds, 
                                   server_means - server_stds, 
                                   server_means + server_stds,
                                   color=server_color, alpha=STD_ALPHA, label='±1 Std Dev')
                
                ax.set_xlabel('Federated Round')
    
    else:
        # For train/val metrics, plot client training curves
        if not plot_rounds or not client_stats:
            return
            
        if show_individual_clients:
            # Plot individual client statistics
            client_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for client_idx, client_id in enumerate(sorted(client_stats.keys())):
                color = client_colors[client_idx % len(client_colors)]
                
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
                        ax.plot(all_steps, all_means, color=color, linewidth=2, 
                               label=f'Client {client_id} (CV avg)', alpha=LINE_ALPHA)
                        
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
                        ax.plot(round_positions, round_means, color=color, marker='o',
                               linewidth=2, markersize=6, label=f'Client {client_id} (CV avg)', 
                               alpha=LINE_ALPHA)
                        
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
                    ax.plot(all_steps, all_means, color='blue', linewidth=2, 
                           label='Cross-Val Average', alpha=LINE_ALPHA)
                    
                    # Plot standard deviation band
                    if show_std and any(std > 0 for std in all_stds):
                        all_means = np.array(all_means)
                        all_stds = np.array(all_stds)
                        ax.fill_between(all_steps, 
                                       all_means - all_stds, 
                                       all_means + all_stds,
                                       color='blue', alpha=STD_ALPHA, label='±1 Std Dev')
                
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
                    ax.plot(round_positions, round_means, color='blue', marker='o',
                           linewidth=2, markersize=6, label='Cross-Val Average', alpha=LINE_ALPHA)
                    
                    # Plot standard deviation error bars
                    if show_std and any(std > 0 for std in round_stds):
                        ax.errorbar(round_positions, round_means, yerr=round_stds,
                                   color='blue', alpha=STD_ALPHA, capsize=5, label='±1 Std Dev')
        
        # Plot server evaluation statistics if available for train/val metrics
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
                # Plot server mean line
                ax.plot(server_rounds, server_means, color='red', marker='s',
                       linewidth=2, markersize=4, linestyle='--', alpha=LINE_ALPHA,
                       label='Global Model (CV avg)')
                
                # Plot server standard deviation error bars
                if show_std and any(std > 0 for std in server_stds):
                    ax.errorbar(server_rounds, server_means, yerr=server_stds,
                               color='red', alpha=STD_ALPHA, capsize=5)
    
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def create_legend_subplot(axs, ax_dims, ax_width, metrics_to_plot, show_individual_clients=False, client_ids=None):
    """Create a dedicated legend subplot"""
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        if base_metric == 'Legend':
            # Get the subplot for legend
            if len(ax_dims) > 1:
                legend_ax = axs[i // ax_width, i % ax_width]
            else:
                legend_ax = axs[i] if hasattr(axs, '__len__') else axs
            
            legend_ax.axis('off')
            
            # Create legend handles
            legend_handles = []
            
            # Add client color legend if showing individual clients
            if show_individual_clients and client_ids:
                client_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                
                # Add client legend entries
                for client_idx, client_id in enumerate(sorted(client_ids)):
                    color = client_colors[client_idx % len(client_colors)]
                    # Use dashed line for test metrics, solid for train/val
                    test_metrics_exist = any('test' in metric for metric in metrics_to_plot.keys() if metric and metric != 'Legend')
                    line_style = '--' if test_metrics_exist else '-'
                    line = plt.Line2D([0], [0], color=color, linewidth=2, linestyle=line_style,
                                     label=f'Client {client_id}')
                    legend_handles.append(line)
            
            # Add server/global model legend
            server_color = 'black' if show_individual_clients and client_ids else 'blue'
            server_marker = 's' if show_individual_clients and client_ids else 'o'
            server_label = 'Global Model (CV avg)' if show_individual_clients and client_ids else 'Cross-Validation Average'
            server_line = plt.Line2D([0], [0], color=server_color, marker=server_marker, 
                                   linewidth=2, markersize=6, label=server_label)
            legend_handles.append(server_line)
            
            legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
            legend_ax.set_title('Cross-Validation Results')
            break


def plot_cross_validation_comparison(experiment_names: list, submodel: str = 'MM', metric_filter: str = 'test',
                                    show_individual_clients: bool = False, show_full_training: bool = False,
                                    smooth_window: int = 0, show_std: bool = True, folds: list = None):
    """
    Compare cross-validation results from multiple federated learning experiments
    
    Note: Updated to handle new test logging format with stage parameter (F1/test/MM/stage).
    Maintains backward compatibility with old format (F1/test/MM).
    
    Args:
        experiment_names: List of experiment names (each should contain fold directories)
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD')
        metric_filter: Which type of metrics to plot ('train', 'val', 'test', or 'all')
        show_individual_clients: If True, show individual client statistics instead of aggregated
        show_full_training: If True, show complete training progress within rounds
        smooth_window: If > 0, apply moving average smoothing with this window size
        show_std: If True, show standard deviation as bands/error bars
        folds: List of specific fold numbers to include (None = use all available)
    """
    
    # Colors for different experiments
    exp_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Get available folds for each experiment
    all_experiment_data = {}
    for exp_name in experiment_names:
        if folds is None:
            available_folds = get_available_folds(exp_name)
            if not available_folds:
                print(f"Warning: No folds found for experiment {exp_name}")
                continue
            fold_numbers = available_folds
        else:
            fold_numbers = folds
        
        print(f"Using folds for {exp_name}: {fold_numbers}")
        all_experiment_data[exp_name] = fold_numbers
    
    if not all_experiment_data:
        raise ValueError("No valid experiments found")
    
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
    exp_names_str = " vs ".join(experiment_names)
    fig.suptitle(f"Cross-Validation Comparison: {exp_names_str}\n({submodel} - {metric_filter} metrics, {title_suffix})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Calculate global parameters from all experiments and folds
    global_all_rounds = set()
    global_max_steps_per_round = 0
    
    for exp_name, fold_numbers in all_experiment_data.items():
        for fold_num in fold_numbers:
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
    
    # Compute statistics for all experiments
    all_experiment_stats = {}
    for exp_name, fold_numbers in all_experiment_data.items():
        print(f"Computing cross-validation statistics for: {exp_name}")
        all_experiment_stats[exp_name] = {}
        
        for base_metric in metrics_to_plot.keys():
            if base_metric == 'Legend':
                continue
            # Use construct_metric_name for consistency, but since we have backward compatibility
            # in tensorboard_to_datadict_federated, we can use the simpler format for now
            metric = construct_metric_name(base_metric, submodel)
            
            client_stats, server_stats = compute_fold_statistics(
                exp_name, fold_numbers, metric, global_all_rounds, global_max_steps_per_round,
                show_full_training, show_individual_clients, smooth_window
            )
            
            all_experiment_stats[exp_name][metric] = (client_stats, server_stats)
    
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
        
        metric = construct_metric_name(base_metric, submodel)
        
        # Plot each experiment's cross-validation results
        for exp_idx, (exp_name, fold_numbers) in enumerate(all_experiment_data.items()):
            if exp_name not in all_experiment_stats or metric not in all_experiment_stats[exp_name]:
                continue
                
            client_stats, server_stats = all_experiment_stats[exp_name][metric]
            color = exp_colors[exp_idx % len(exp_colors)]
            
            # Use a modified plot function that accepts a color parameter
            plot_crossval_metric_with_color(ax, metric, title, client_stats, server_stats, global_all_rounds,
                                           global_max_steps_per_round, show_full_training, show_individual_clients,
                                           show_std, show_legend=not has_legend_entry, color=color, 
                                           label_prefix=exp_name)
    
    # Create comparison legend if needed
    if has_legend_entry:
        comparison_legend_handles = []
        for exp_idx, exp_name in enumerate(all_experiment_data.keys()):
            color = exp_colors[exp_idx % len(exp_colors)]
            line = plt.Line2D([0], [0], color=color, linewidth=2, label=f'{exp_name} (CV avg)')
            comparison_legend_handles.append(line)
        
        # Add general legend elements (removed standard deviation and global model info as they're clear from context)
        
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
    exp_names_clean = "_vs_".join([exp.replace("_", "-") for exp in experiment_names])
    plot_path = os.path.join(ROOT_RESULTS, f"crossval_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Cross-validation comparison plot saved to {plot_path}")
    plt.show()


def plot_crossval_metric_with_color(ax, metric, title, client_stats, server_stats, all_rounds, 
                                   global_max_steps_per_round, show_full_training, show_individual_clients, 
                                   show_std=True, show_legend=True, color='blue', label_prefix=''):
    """
    Plot a single metric for cross-validation results with specified color and label prefix
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
        # For test metrics, show individual clients if requested, otherwise show server evaluation
        if show_individual_clients and client_stats:
            # Plot individual client test statistics with different line styles
            client_line_styles = ['-', '--', '-.', ':']
            
            for client_idx, client_id in enumerate(sorted(client_stats.keys())):
                client_line_style = client_line_styles[client_idx % len(client_line_styles)]
                
                # For test metrics, we only have final values (no full training curves)
                client_means = []
                client_stds = []
                client_rounds = []
                
                for round_num in sorted(client_stats[client_id].keys()):
                    if round_num in plot_rounds:
                        mean_val = client_stats[client_id][round_num]['mean']
                        std_val = client_stats[client_id][round_num]['std']
                        
                        # Handle both single values and lists
                        if isinstance(mean_val, list):
                            mean_val = mean_val[-1] if mean_val else 0.0  # Take last value if list
                        if isinstance(std_val, list):
                            std_val = std_val[-1] if std_val else 0.0    # Take last value if list
                            
                        client_means.append(mean_val)
                        client_stds.append(std_val)
                        client_rounds.append(round_num)
                
                if client_means:
                    # Plot client mean line with dashed style (override client_line_style for test metrics)
                    client_label = f'{label_prefix} C{client_id} (CV avg)' if label_prefix else f'Client {client_id} (CV avg)'
                    ax.plot(client_rounds, client_means, color=color, marker='o', linestyle='--',
                           linewidth=2, markersize=4, label=client_label, alpha=LINE_ALPHA)
                    
                    # Plot standard deviation shaded band for individual clients
                    if show_std and client_stds and any(std > 0 for std in client_stds):
                        client_means_array = np.array(client_means)
                        client_stds_array = np.array(client_stds)
                        ax.fill_between(client_rounds, 
                                       client_means_array - client_stds_array, 
                                       client_means_array + client_stds_array,
                                       color=color, alpha=STD_ALPHA * 0.6)  # Lighter shade than server
            
            ax.set_xlabel('Federated Round')
        
        # Always show server evaluation for test metrics
        if server_stats:
            # Default behavior: plot server evaluation points only
            server_means = []
            server_stds = []
            server_rounds = []
            
            for round_num in sorted(server_stats.keys()):
                if round_num in plot_rounds:
                    server_means.append(server_stats[round_num]['mean'])
                    server_stds.append(server_stats[round_num]['std'])
                    server_rounds.append(round_num)
            
            if server_means:
                # Plot mean line - use square markers to distinguish from clients when showing individual clients
                server_marker = 's' if show_individual_clients and client_stats else 'o'
                server_label_suffix = ' (Global)' if show_individual_clients and client_stats else ' (CV avg)'
                label = f'{label_prefix}{server_label_suffix}' if label_prefix else 'Cross-Val Average'
                ax.plot(server_rounds, server_means, color=color, marker=server_marker, 
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
                        label = f'{label_prefix} C{client_id} (CV avg)' if label_prefix else f'Client {client_id} (CV avg)'
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
                        label = f'{label_prefix} C{client_id} (CV avg)' if label_prefix else f'Client {client_id} (CV avg)'
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
                    label = f'{label_prefix} (CV avg)' if label_prefix else 'Cross-Val Average'
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
                    label = f'{label_prefix} (CV avg)' if label_prefix else 'Cross-Val Average'
                    ax.plot(round_positions, round_means, color=color, marker='o',
                           linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA)
                    
                    # Plot standard deviation error bars
                    if show_std and any(std > 0 for std in round_stds):
                        ax.errorbar(round_positions, round_means, yerr=round_stds,
                                   color=color, alpha=STD_ALPHA, capsize=5)
        
        # Plot server evaluation statistics if available for train/val metrics
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
                # Plot server mean line
                server_label = f'{label_prefix} (Global CV avg)' if label_prefix else 'Global Model (CV avg)'
                ax.plot(server_rounds, server_means, color=color, marker='s',
                       linewidth=2, markersize=4, linestyle='--', alpha=LINE_ALPHA,
                       label=server_label)
                
                # Plot server standard deviation error bars
                if show_std and any(std > 0 for std in server_stds):
                    ax.errorbar(server_rounds, server_means, yerr=server_stds,
                               color=color, alpha=STD_ALPHA, capsize=5)
    
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


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
            legend_ax.set_title('Cross-Validation Comparison')
            break


def plot_cross_validation(experiment_name: str, submodel: str = 'MM', metric_filter: str = 'test',
                         show_individual_clients: bool = False, show_full_training: bool = False,
                         smooth_window: int = 0, show_std: bool = True, folds: list = None):
    """
    Plot cross-validation results for a federated learning experiment
    
    Note: Updated to handle new test logging format with stage parameter (F1/test/MM/stage).
    Maintains backward compatibility with old format (F1/test/MM).
    
    Args:
        experiment_name: Name of the experiment (should contain fold directories)
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD')
        metric_filter: Which type of metrics to plot ('train', 'val', 'test', or 'all')
        show_individual_clients: If True, show individual client statistics instead of aggregated
        show_full_training: If True, show complete training progress within rounds
        smooth_window: If > 0, apply moving average smoothing with this window size
        show_std: If True, show standard deviation as bands/error bars
        folds: List of specific fold numbers to include (None = use all available)
    """
    
    # Get available folds
    if folds is None:
        available_folds = get_available_folds(experiment_name)
        if not available_folds:
            raise ValueError(f"No folds found for experiment {experiment_name}")
        fold_numbers = available_folds
    else:
        fold_numbers = folds
    
    print(f"Using folds: {fold_numbers}")
    
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
    fig.suptitle(f"Cross-Validation Results: {experiment_name}\n({submodel} - {metric_filter} metrics, {len(fold_numbers)} folds, {title_suffix})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Calculate global parameters from all folds
    global_all_rounds = set()
    global_max_steps_per_round = 0
    
    for fold_num in fold_numbers:
        try:
            client_data, server_data = tensorboard_to_datadict_federated(experiment_name, fold_num)
            for client_id in client_data.keys():
                global_all_rounds.update(client_data[client_id].keys())
                for round_num in client_data[client_id].keys():
                    for metric_key in client_data[client_id][round_num].keys():
                        if client_data[client_id][round_num][metric_key]:
                            steps_in_round = len([v for v in client_data[client_id][round_num][metric_key] if v is not None])
                            global_max_steps_per_round = max(global_max_steps_per_round, steps_in_round)
            global_all_rounds.update(server_data.keys())
        except Exception as e:
            print(f"Warning: Could not load data for fold {fold_num}: {e}")
            continue
    
    global_all_rounds = sorted(global_all_rounds)
    
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
        
        metric = construct_metric_name(base_metric, submodel)
        
        # Compute statistics for this metric
        client_stats, server_stats = compute_fold_statistics(
            experiment_name, fold_numbers, metric, global_all_rounds, global_max_steps_per_round,
            show_full_training, show_individual_clients, smooth_window
        )
        
        plot_crossval_metric(ax, metric, title, client_stats, server_stats, global_all_rounds,
                           global_max_steps_per_round, show_full_training, show_individual_clients,
                           show_std, show_legend=not has_legend_entry)
    
    # Create centralized legend if needed
    if has_legend_entry:
        # Collect client information for legend
        all_client_ids = set()
        if show_individual_clients:
            for fold_num in fold_numbers:
                try:
                    client_data, _ = tensorboard_to_datadict_federated(experiment_name, fold_num)
                    all_client_ids.update(client_data.keys())
                except:
                    continue
        
        create_legend_subplot(axs, ax_dims, ax_width, metrics_to_plot, show_individual_clients, all_client_ids)
    
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
    plot_path = os.path.join(ROOT_RESULTS, f"crossval_{experiment_name}_{submodel.lower()}_{metric_filter}_{suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Cross-validation plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Validation Plotting Script for Federated Learning')
    parser.add_argument('--name', type=str,
                       help='Name of the experiment (should contain fold directories)')
    parser.add_argument('--names', nargs='+',
                       help='Names of experiments to compare (each should contain fold directories)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple cross-validation experiments')
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
    
    # Handle std flag logic
    show_std = args.show_std and not args.no_std
    
    # Use comparison if requested
    if args.compare and args.names:
        plot_cross_validation_comparison(args.names, args.submodel, args.metric_filter,
                                       args.show_individual_clients, args.show_full_training,
                                       args.smooth_window, show_std, args.folds)
    elif args.name:
        plot_cross_validation(args.name, args.submodel, args.metric_filter,
                             args.show_individual_clients, args.show_full_training,
                             args.smooth_window, show_std, args.folds)
    else:
        parser.error("Either --name (for single experiment) or --compare --names (for comparison) is required")

# Example usage:

# Single experiment cross-validation:
# python plot_cross_val.py --name folds_test_s1 --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# Show only specific folds:
# python plot_cross_val.py --name folds_test_s1 --submodel MM --metric_filter test --folds 0 1 2

# Show individual clients:
# python plot_cross_val.py --name folds_test_s1 --submodel MM --metric_filter all --show_individual_clients --show_std

# Compare cross-validation results between experiments:
# python plot_cross_val.py --compare --names folds_3_clients_no_overfit_sp1_4115_redo_s1 folds_3_clients_no_overfit_sp1_4115_redo_s2 --submodel MM --metric_filter test --show_std

# Compare with all metrics and full training curves:
# python plot_cross_val.py --compare --names folds_3_clients_no_overfit_sp1_4115_redo_s1 folds_3_clients_no_overfit_sp1_4115_redo_s2 --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# cd /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/plot && conda activate osr && python plot_cross_val.py --compare --names folds_3_clients_no_overfit_sp1_4115_redo_s1 folds_3_clients_no_overfit_sp1_4115_redo_s2 --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

#python plot_cross_val.py --compare --names no_phases_s1 no_phases_s2 no_phases_s3 no_phases_s4 no_phases_s5 --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std