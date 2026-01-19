import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
from collections import defaultdict
import argparse
import numpy as np

ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/results"

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


def tensorboard_to_datadict_federated(experiment_name: str, exp_dir: str = ROOT_RESULTS):
    """Extract data from TensorBoard logs for federated learning plotting"""
    
    experiment_dir = os.path.join(exp_dir, experiment_name)
    log_dir = os.path.join(experiment_dir, "log")
    
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."
    
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
            for file in os.listdir(client_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(client_path, file)
                    for event in tf.compat.v1.train.summary_iterator(event_file):
                        for value in event.summary.value:
                            # Store all epochs for this client/round/metric
                            metric = value.tag
                            while len(client_data[client_id][round_num][metric]) <= event.step:
                                client_data[client_id][round_num][metric].append(None)
                            client_data[client_id][round_num][metric][event.step] = value.simple_value

        elif "server" in item:
            # Parse server evaluation logs: client_server_{round}
            round_num = int(item.split("_")[2])
            
            server_path = os.path.join(log_dir, item)
            for file in os.listdir(server_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(server_path, file)
                    for event in tf.compat.v1.train.summary_iterator(event_file):
                        for value in event.summary.value:
                            # Server evaluation metrics (single values per round)
                            server_data[round_num][value.tag] = value.simple_value
    
    return dict(client_data), dict(server_data)


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


def compute_group_statistics(group_experiments, metric, all_rounds, global_max_steps_per_round, 
                           show_full_training, show_individual_clients, smooth_window=0):
    """
    Compute mean and standard deviation statistics across experiments in a group
    
    Args:
        group_experiments: List of experiment names in the group
        metric: The metric to analyze
        all_rounds: List of rounds to consider
        global_max_steps_per_round: Maximum steps per round for consistent spacing
        show_full_training: Whether to show full training curves or just final values
        show_individual_clients: Whether to compute stats per client or aggregated
        smooth_window: Window size for smoothing (0 = no smoothing)
        
    Returns:
        Tuple of (mean_data, std_data, server_mean, server_std)
    """
    # Load data for all experiments in the group
    group_data = []
    for exp_name in group_experiments:
        try:
            client_data, server_data = tensorboard_to_datadict_federated(exp_name)
            
            # Apply smoothing if requested
            if smooth_window > 0:
                client_data = smooth_client_data(client_data, smooth_window)
            
            group_data.append((client_data, server_data))
        except Exception as e:
            print(f"Warning: Could not load data for {exp_name}: {e}")
            continue
    
    if not group_data:
        return None, None, None, None
    
    # Compute server statistics (simpler case)
    server_stats = {}
    for round_num in all_rounds:
        round_values = []
        for _, server_data in group_data:
            if round_num in server_data and metric in server_data[round_num]:
                round_values.append(server_data[round_num][metric])
        
        if round_values:
            server_stats[round_num] = {
                'mean': np.mean(round_values),
                'std': np.std(round_values, ddof=1) if len(round_values) > 1 else 0.0
            }
    
    # Compute client statistics
    if show_individual_clients:
        # Compute statistics per client ID
        client_stats = {}
        
        # Get all client IDs across all experiments
        all_client_ids = set()
        for client_data, _ in group_data:
            all_client_ids.update(client_data.keys())
        
        for client_id in sorted(all_client_ids):
            client_stats[client_id] = {}
            
            if show_full_training:
                # Statistics for full training curves
                for round_idx, round_num in enumerate(all_rounds):
                    # Collect data for this client/round across all experiments
                    experiments_data = []
                    max_steps_this_round = 0
                    
                    for client_data, _ in group_data:
                        if client_id in client_data and round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if client_round_values:
                                experiments_data.append(client_round_values)
                                max_steps_this_round = max(max_steps_this_round, len(client_round_values))
                    
                    if experiments_data and max_steps_this_round > 0:
                        # Align all experiments to same length (pad with last value)
                        aligned_data = []
                        for exp_values in experiments_data:
                            padded_values = exp_values + [exp_values[-1]] * (max_steps_this_round - len(exp_values))
                            aligned_data.append(padded_values)
                        
                        # Compute statistics for each step
                        mean_curve = []
                        std_curve = []
                        
                        for step in range(max_steps_this_round):
                            step_values = [exp_data[step] for exp_data in aligned_data]
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
                    
                    for client_data, _ in group_data:
                        if client_id in client_data and round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if client_round_values:
                                final_values.append(client_round_values[-1])
                    
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
                # Collect aggregated values for this round across all experiments
                experiments_aggregated = []
                max_steps_this_round = 0
                
                for client_data, _ in group_data:
                    # Aggregate all clients for this experiment/round
                    round_client_data = []
                    
                    for client_id in sorted(client_data.keys()):
                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if client_round_values:
                                round_client_data.append(client_round_values)
                                max_steps_this_round = max(max_steps_this_round, len(client_round_values))
                    
                    if round_client_data:
                        # Average across clients for each step in this round
                        avg_curve = []
                        for step in range(max_steps_this_round):
                            step_values = []
                            for client_values in round_client_data:
                                if step < len(client_values):
                                    step_values.append(client_values[step])
                            
                            if step_values:
                                avg_curve.append(np.mean(step_values))
                        
                        if avg_curve:
                            experiments_aggregated.append(avg_curve)
                
                if experiments_aggregated and max_steps_this_round > 0:
                    # Align all experiments to same length
                    aligned_data = []
                    for exp_curve in experiments_aggregated:
                        padded_curve = exp_curve + [exp_curve[-1]] * (max_steps_this_round - len(exp_curve))
                        aligned_data.append(padded_curve)
                    
                    # Compute statistics for each step
                    mean_curve = []
                    std_curve = []
                    
                    for step in range(max_steps_this_round):
                        step_values = [exp_data[step] for exp_data in aligned_data]
                        mean_curve.append(np.mean(step_values))
                        std_curve.append(np.std(step_values, ddof=1) if len(step_values) > 1 else 0.0)
                    
                    aggregated_stats[round_num] = {
                        'mean': mean_curve,
                        'std': std_curve
                    }
        
        else:
            # Statistics for aggregated final values
            for round_num in all_rounds:
                final_aggregated_values = []
                
                for client_data, _ in group_data:
                    # Collect final values from all clients for this experiment/round
                    round_final_values = []
                    
                    for client_id in sorted(client_data.keys()):
                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                            if client_round_values:
                                round_final_values.append(client_round_values[-1])
                    
                    if round_final_values:
                        # Average across clients for this experiment
                        final_aggregated_values.append(np.mean(round_final_values))
                
                if final_aggregated_values:
                    aggregated_stats[round_num] = {
                        'mean': np.mean(final_aggregated_values),
                        'std': np.std(final_aggregated_values, ddof=1) if len(final_aggregated_values) > 1 else 0.0
                    }
        
        client_stats = aggregated_stats
    
    return client_stats, server_stats


def plot_group_metric(ax, metric, title, group_stats, group_colors, group_names, all_rounds, 
                     global_max_steps_per_round, show_full_training, show_individual_clients, 
                     show_std=True, show_legend=True, show_individual_experiments=False, 
                     experiment_groups=None):
    """
    Plot a single metric comparison across experiment groups
    
    Args:
        ax: Matplotlib axis to plot on
        metric: The metric being plotted
        title: Title for the plot
        group_stats: Dict mapping group names to their statistics
        group_colors: List of colors for groups
        group_names: List of group names in order
        all_rounds: List of rounds to plot
        global_max_steps_per_round: Maximum steps per round for spacing
        show_full_training: Whether showing full training curves
        show_individual_clients: Whether showing individual clients
        show_std: Whether to show standard deviation bands
        show_legend: Whether to show legend
        show_individual_experiments: Whether to show individual experiments as light background curves
        experiment_groups: Dict mapping group names to experiment lists (needed for individual experiments)
    """
    
    # Alpha (transparency) settings
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.3
    INDIVIDUAL_EXP_ALPHA = 0.15  # Light alpha for individual experiment curves
    
    # For train/val metrics, exclude round 0 (initial evaluation before training)
    if 'test' not in metric:
        plot_rounds = [r for r in all_rounds if r > 0]
    else:
        plot_rounds = all_rounds
    
    # Plot individual experiments as light background curves if requested
    if show_individual_experiments and experiment_groups:
        for group_idx, group_name in enumerate(group_names):
            if group_name not in experiment_groups:
                continue
                
            color = group_colors[group_idx % len(group_colors)]
            group_experiments = experiment_groups[group_name]
            
            for exp_idx, exp_name in enumerate(group_experiments):
                try:
                    client_data, server_data = tensorboard_to_datadict_federated(exp_name)
                    
                    if 'test' in metric:
                        # For test metrics, plot individual server evaluation points
                        server_values = []
                        server_rounds = []
                        for round_num in sorted(server_data.keys()):
                            if round_num in plot_rounds and metric in server_data[round_num]:
                                server_values.append(server_data[round_num][metric])
                                server_rounds.append(round_num)
                        
                        if server_values:
                            ax.plot(server_rounds, server_values, color=color, 
                                   linewidth=1, alpha=INDIVIDUAL_EXP_ALPHA, linestyle='-')
                    
                    else:
                        # For train/val metrics, plot individual client curves
                        if show_individual_clients:
                            # Plot each client for this experiment
                            for client_id in sorted(client_data.keys()):
                                if show_full_training:
                                    # Plot full training curve for this client
                                    all_values = []
                                    all_steps = []
                                    max_steps_per_round = global_max_steps_per_round
                                    
                                    for round_idx, round_num in enumerate(plot_rounds):
                                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                            round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                            
                                            if round_values:
                                                round_start = round_idx * max_steps_per_round
                                                round_steps = list(range(round_start, round_start + len(round_values)))
                                                all_values.extend(round_values)
                                                all_steps.extend(round_steps)
                                    
                                    if all_values:
                                        ax.plot(all_steps, all_values, color=color, 
                                               linewidth=1, alpha=INDIVIDUAL_EXP_ALPHA, linestyle='-')
                                
                                else:
                                    # Plot final values for this client
                                    round_values = []
                                    round_positions = []
                                    
                                    for round_num in plot_rounds:
                                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                            if client_round_values:
                                                round_values.append(client_round_values[-1])
                                                round_positions.append(round_num)
                                    
                                    if round_values:
                                        ax.plot(round_positions, round_values, color=color, 
                                               linewidth=1, alpha=INDIVIDUAL_EXP_ALPHA, linestyle='-')
                        
                        else:
                            # Plot aggregated curve for this experiment
                            if show_full_training:
                                # Plot full aggregated training curve
                                all_avg_values = []
                                all_steps = []
                                max_steps_per_round = global_max_steps_per_round
                                
                                for round_idx, round_num in enumerate(plot_rounds):
                                    # Aggregate all clients for this round
                                    round_client_data = []
                                    max_steps_this_round = 0
                                    
                                    for client_id in sorted(client_data.keys()):
                                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                            if client_round_values:
                                                round_client_data.append(client_round_values)
                                                max_steps_this_round = max(max_steps_this_round, len(client_round_values))
                                    
                                    if round_client_data:
                                        # Average across clients for each step
                                        round_start = round_idx * max_steps_per_round
                                        for step in range(max_steps_this_round):
                                            step_values = []
                                            for client_values in round_client_data:
                                                if step < len(client_values):
                                                    step_values.append(client_values[step])
                                            
                                            if step_values:
                                                avg_value = sum(step_values) / len(step_values)
                                                all_avg_values.append(avg_value)
                                                all_steps.append(round_start + step)
                                
                                if all_avg_values:
                                    ax.plot(all_steps, all_avg_values, color=color, 
                                           linewidth=1, alpha=INDIVIDUAL_EXP_ALPHA, linestyle='-')
                            
                            else:
                                # Plot aggregated final values
                                round_aggregated_values = []
                                round_positions = []
                                
                                for round_num in plot_rounds:
                                    round_final_values = []
                                    
                                    for client_id in sorted(client_data.keys()):
                                        if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                            client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                            if client_round_values:
                                                round_final_values.append(client_round_values[-1])
                                    
                                    if round_final_values:
                                        avg_value = sum(round_final_values) / len(round_final_values)
                                        round_aggregated_values.append(avg_value)
                                        round_positions.append(round_num)
                                
                                if round_aggregated_values:
                                    ax.plot(round_positions, round_aggregated_values, color=color, 
                                           linewidth=1, alpha=INDIVIDUAL_EXP_ALPHA, linestyle='-')
                
                except Exception as e:
                    print(f"Warning: Could not plot individual experiment {exp_name}: {e}")
                    continue
    
    for group_idx, group_name in enumerate(group_names):
        if group_name not in group_stats:
            continue
            
        client_stats, server_stats = group_stats[group_name]
        color = group_colors[group_idx % len(group_colors)]
        
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
                    ax.plot(server_rounds, server_means, color=color, marker='o', 
                           linewidth=2, markersize=6, label=group_name, alpha=LINE_ALPHA)
                    
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
                continue
                
            if show_individual_clients:
                # Plot individual client statistics
                for client_idx, client_id in enumerate(sorted(client_stats.keys())):
                    client_line_style = ['-', '--', '-.', ':'][client_idx % 4]
                    
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
                                
                                if round_means:
                                    round_start = round_idx * max_steps_per_round
                                    round_steps = list(range(round_start, round_start + len(round_means)))
                                    
                                    all_means.extend(round_means)
                                    all_stds.extend(round_stds)
                                    all_steps.extend(round_steps)
                        
                        if all_means:
                            # Plot mean curve
                            ax.plot(all_steps, all_means, color=color, linestyle=client_line_style,
                                   linewidth=1.5, alpha=LINE_ALPHA, 
                                   label=f'{group_name} C{client_id}')
                            
                            # Plot standard deviation band
                            if show_std and any(std > 0 for std in all_stds):
                                all_means = np.array(all_means)
                                all_stds = np.array(all_stds)
                                ax.fill_between(all_steps,
                                               all_means - all_stds,
                                               all_means + all_stds,
                                               color=color, alpha=STD_ALPHA)
                        
                        # Set x-axis labels
                        if plot_rounds and max_steps_per_round > 0:
                            round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(plot_rounds))]
                            ax.set_xticks(round_centers)
                            ax.set_xticklabels([f'R{r}' for r in plot_rounds])
                            ax.set_xlabel('Federated Learning Progress')
                    
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
                            ax.plot(round_positions, round_means, color=color, 
                                   linestyle=client_line_style, linewidth=2, markersize=4,
                                   alpha=LINE_ALPHA, label=f'{group_name} C{client_id}')
                            
                            # Plot standard deviation error bars
                            if show_std and any(std > 0 for std in round_stds):
                                ax.errorbar(round_positions, round_means, yerr=round_stds,
                                           color=color, alpha=STD_ALPHA, linestyle='none', capsize=3)
                            
                            ax.set_xlabel('Federated Round')
            
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
                            
                            if round_means:
                                round_start = round_idx * max_steps_per_round
                                round_steps = list(range(round_start, round_start + len(round_means)))
                                
                                all_means.extend(round_means)
                                all_stds.extend(round_stds)
                                all_steps.extend(round_steps)
                    
                    if all_means:
                        # Plot mean curve
                        ax.plot(all_steps, all_means, color=color, linewidth=2,
                               alpha=LINE_ALPHA, label=f'{group_name} (avg)')
                        
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
                            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
                    
                    # Set x-axis labels
                    if plot_rounds and max_steps_per_round > 0:
                        round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(plot_rounds))]
                        ax.set_xticks(round_centers)
                        ax.set_xticklabels([f'R{r}' for r in plot_rounds])
                        ax.set_xlabel('Federated Learning Progress')
                
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
                        ax.plot(round_positions, round_means, color=color, marker='o',
                               linewidth=2, markersize=6, alpha=LINE_ALPHA, 
                               label=f'{group_name} (avg)')
                        
                        # Plot standard deviation error bars
                        if show_std and any(std > 0 for std in round_stds):
                            ax.errorbar(round_positions, round_means, yerr=round_stds,
                                       color=color, alpha=STD_ALPHA, linestyle='none', capsize=5)
                        
                        ax.set_xlabel('Federated Round')
            
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
                    ax.plot(server_rounds, server_means, color=color, marker='s',
                           linewidth=2, markersize=4, linestyle='--', alpha=LINE_ALPHA,
                           label=f'{group_name} (server)')
                    
                    # Plot server standard deviation error bars
                    if show_std and any(std > 0 for std in server_stds):
                        ax.errorbar(server_rounds, server_means, yerr=server_stds,
                                   color=color, alpha=STD_ALPHA, linestyle='none', capsize=3)
    
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, ncol=1 if len(handles) <= 6 else 2)


def create_group_legend_handles(group_names, group_colors):
    """Create legend handles for the groups"""
    legend_handles = []
    for group_idx, group_name in enumerate(group_names):
        color = group_colors[group_idx % len(group_colors)]
        line = plt.Line2D([0], [0], color=color, linewidth=2, label=group_name)
        legend_handles.append(line)
    return legend_handles


def create_legend_subplot(axs, ax_dims, ax_width, legend_handles):
    """Create a dedicated legend subplot"""
    for i, (base_metric, title) in enumerate(federated_metrics.items()):
        if base_metric == 'Legend':
            # Get the subplot for legend
            if len(ax_dims) > 1:
                legend_ax = axs[i // ax_width, i % ax_width]
            else:
                legend_ax = axs[i] if hasattr(axs, '__len__') else axs
            
            legend_ax.axis('off')
            legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
            legend_ax.set_title('Experiment Groups')
            break


def plot_group_comparison(experiment_groups: dict, submodel: str = 'MM', metric_filter: str = 'test',
                         show_individual_clients: bool = False, show_full_training: bool = False,
                         smooth_window: int = 0, show_std: bool = True, show_individual_experiments: bool = False):
    """
    Compare experiment groups with averaged results and optional standard deviation
    
    Args:
        experiment_groups: Dict mapping group names to lists of experiment names
                          e.g., {'Group1': ['exp1_s1', 'exp1_s2'], 'Group2': ['exp2_s1', 'exp2_s2']}
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD')
        metric_filter: Which type of metrics to plot ('train', 'val', 'test', or 'all')
        show_individual_clients: If True, show individual client statistics instead of aggregated
        show_full_training: If True, show complete training progress within rounds
        smooth_window: If > 0, apply moving average smoothing with this window size
        show_std: If True, show standard deviation as bands/error bars
        show_individual_experiments: If True, show individual experiments as light background curves
    """
    
    # Filter metrics based on metric_filter
    if metric_filter == 'all':
        metrics_to_plot = {k: v for k, v in federated_metrics.items() if k is not None and v is not None}
    else:
        metrics_to_plot = {k: v for k, v in federated_metrics.items() 
                          if k is not None and v is not None and (metric_filter in k or k == 'Legend')}
    
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
    if show_individual_experiments:
        mode_desc.append("with individual experiments")
    
    title_suffix = ", ".join(mode_desc)
    group_names_str = " vs ".join(experiment_groups.keys())
    fig.suptitle(f"Group Comparison: {group_names_str}\n({submodel} - {metric_filter} metrics, {title_suffix})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Colors for different groups
    group_colors = [
        'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'lime', 'indigo', 'teal', 'maroon', 'navy'
    ]
    
    group_names = list(experiment_groups.keys())
    
    # Calculate global parameters from all experiments
    all_experiments = [exp for group_exps in experiment_groups.values() for exp in group_exps]
    global_all_rounds = set()
    global_max_steps_per_round = 0
    
    for exp_name in all_experiments:
        try:
            client_data, server_data = tensorboard_to_datadict_federated(exp_name)
            for client_id in client_data.keys():
                global_all_rounds.update(client_data[client_id].keys())
                for round_num in client_data[client_id].keys():
                    for metric_key in client_data[client_id][round_num].keys():
                        if client_data[client_id][round_num][metric_key]:
                            steps_in_round = len([v for v in client_data[client_id][round_num][metric_key] if v is not None])
                            global_max_steps_per_round = max(global_max_steps_per_round, steps_in_round)
            global_all_rounds.update(server_data.keys())
        except Exception as e:
            print(f"Warning: Could not load data for {exp_name}: {e}")
            continue
    
    global_all_rounds = sorted(global_all_rounds)
    
    # Compute statistics for all groups
    group_stats = {}
    for group_name, group_experiments in experiment_groups.items():
        print(f"Computing statistics for group: {group_name}")
        for base_metric in metrics_to_plot.keys():
            if base_metric == 'Legend':
                continue
            metric = f"{base_metric}/{submodel}"
            
            client_stats, server_stats = compute_group_statistics(
                group_experiments, metric, global_all_rounds, global_max_steps_per_round,
                show_full_training, show_individual_clients, smooth_window
            )
            
            if group_name not in group_stats:
                group_stats[group_name] = {}
            group_stats[group_name][metric] = (client_stats, server_stats)
    
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
        
        # Collect statistics for this metric from all groups
        metric_group_stats = {}
        for group_name in group_names:
            if group_name in group_stats and metric in group_stats[group_name]:
                metric_group_stats[group_name] = group_stats[group_name][metric]
        
        plot_group_metric(ax, metric, title, metric_group_stats, group_colors, group_names,
                         global_all_rounds, global_max_steps_per_round, show_full_training,
                         show_individual_clients, show_std, show_legend=not has_legend_entry,
                         show_individual_experiments=show_individual_experiments, 
                         experiment_groups=experiment_groups)
    
    # Create centralized legend if needed
    if has_legend_entry:
        legend_handles = create_group_legend_handles(group_names, group_colors)
        create_legend_subplot(axs, ax_dims, ax_width, legend_handles)
    
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
    
    if show_individual_experiments:
        suffix_parts.append("withindividuals")
    
    suffix = "_".join(suffix_parts)
    plot_path = os.path.join(ROOT_RESULTS, f"group_comparison_{submodel.lower()}_{metric_filter}_{suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Group comparison plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Group Comparison Plotting Script')
    parser.add_argument('--groups', type=str, required=True,
                       help='JSON string or file path defining experiment groups. Format: {"Group1": ["exp1", "exp2"], "Group2": ["exp3", "exp4"]}')
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
    parser.add_argument('--show_individual_experiments', action='store_true',
                       help='Show individual experiments as light background curves behind group averages')
    
    args = parser.parse_args()
    
    # Handle std flag logic
    show_std = args.show_std and not args.no_std
    
    # Parse groups argument
    import json
    try:
        # Try to parse as JSON string first
        experiment_groups = json.loads(args.groups)
    except json.JSONDecodeError:
        # Try to read as file path
        try:
            with open(args.groups, 'r') as f:
                experiment_groups = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: Could not parse groups argument '{args.groups}' as JSON string or file path")
            exit(1)
    
    plot_group_comparison(experiment_groups, args.submodel, args.metric_filter,
                         args.show_individual_clients, args.show_full_training,
                         args.smooth_window, show_std, args.show_individual_experiments)

# Example usage:
# python plot_group_comparison.py --groups '{"Seeds_1-3": ["3_clients_no_overfit_sp1_4115_s1", "3_clients_no_overfit_sp1_4115_s2", "3_clients_no_overfit_sp1_4115_s3"], "Seeds_4-5": ["3_clients_no_overfit_sp1_4115_s4", "3_clients_no_overfit_sp1_4115_s5"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 10 --show_std --show_individual_experiments

# Or with a JSON file:
# echo '{"Group_A": ["exp1_s1", "exp1_s2", "exp1_s3"], "Group_B": ["exp2_s1", "exp2_s2", "exp2_s3"]}' > groups.json
# python plot_group_comparison.py --groups example_groups.json --show_std --show_individual_experiments