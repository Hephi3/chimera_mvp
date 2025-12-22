import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results"
OLD_DATA_ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results_old_data"

# ROOT_RESULTS = "/home/phempel/tmp_filerworkaround_phempel_nov_2025/train_cfcd/train_cfcd/results"
# OLD_DATA_ROOT_RESULTS = "/home/phempel/tmp_filerworkaround_phempel_nov_2025/train_cfcd/train_cfcd/results_old"


def calculate_avg_f1_metric(f1_s1test_s1, f1_s1test_s2, f1_s2test_s1, f1_s2test_s2, plasticity_weight=0.5):
    '''
    Calculate the avg_f1 metric for continual learning evaluation.
    
    Parameters:
        - f1_s1test_s1: F1 Score of the stage 1 testset at the end of stage 1
        - f1_s1test_s2: F1 Score of the stage 1 testset at stage 2
        - f1_s2test_s1: F1 Score of the stage 2 testset at the end of stage 1
        - f1_s2test_s2: F1 Score of the stage 2 testset at stage 2
        
    Measures plasticity and performance retention of CF setups:
        - Plasticity: Improvement on stage 2 testset from stage 1 to stage 2
        - Retention: Change on stage 1 testset from stage 1 to stage 2
    '''
    retention_weight = 1.0 - plasticity_weight
    return retention_weight * f1_s1test_s2 + plasticity_weight * f1_s2test_s2 - (retention_weight * f1_s1test_s1 + plasticity_weight * f1_s2test_s1)


def calculate_avg_f1_metric(f1_s1test_s1, f1_s1test_s2, f1_s2test_s1, f1_s2test_s2, plasticity_weight=0.5):
    '''
    Calculate the avg_f1 metric for continual learning evaluation.
    
    Parameters:
        - f1_s1test_s1: F1 Score of the stage 1 testset at the end of stage 1
        - f1_s1test_s2: F1 Score of the stage 1 testset at stage 2
        - f1_s2test_s1: F1 Score of the stage 2 testset at the end of stage 1
        - f1_s2test_s2: F1 Score of the stage 2 testset at stage 2
        
    Measures plasticity and performance retention of CF setups:
        - Plasticity: Improvement on stage 2 testset from stage 1 to stage 2
        - Retention: Change on stage 1 testset from stage 1 to stage 2
    '''
    retention_weight = 1.0 - plasticity_weight
    return retention_weight * f1_s1test_s2 + plasticity_weight * f1_s2test_s2 - (retention_weight * f1_s1test_s1 + plasticity_weight * f1_s2test_s1)

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
    # 'Accuracy/test': 'Test Accuracy',
    # 'Binary_Accuracy/test': 'Test Binary Accuracy',
    # 'ROC_AUC/test': 'Test ROC AUC',
    'Avg_f1/test': 'Test Average F1',
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

        elif "server_" in item:
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
        elif "server2" in item:
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


def plot_crossfold_group_test_metric(ax, metric, title, client_stats, server_stats, all_rounds, 
                                    global_max_steps_per_round, show_full_training, show_individual_clients, 
                                    show_std=True, show_legend=True, color='blue', label_prefix=''):
    """
    Plot test metrics with stage differentiation for cross-fold group comparison results
    """
    
    # Alpha (transparency) settings
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.12  # Reduced for better visibility
    
    plot_rounds = all_rounds
    
    if server_stats:
        # Extract stage data from combined server stats
        stage_0_rounds = []
        stage_1_rounds = []
        stage_0_means = []
        stage_1_means = []
        stage_0_stds = []
        stage_1_stds = []
        
        for round_num in sorted(server_stats.keys()):
            if round_num in plot_rounds:
                round_data = server_stats[round_num]
                
                # Check for Stage 0 data
                if 'stage_0' in round_data:
                    stage_0_rounds.append(round_num)
                    stage_0_means.append(round_data['stage_0']['mean'])
                    stage_0_stds.append(round_data['stage_0']['std'])
                
                # Check for Stage 1 data
                if 'stage_1' in round_data:
                    stage_1_rounds.append(round_num)
                    stage_1_means.append(round_data['stage_1']['mean'])
                    stage_1_stds.append(round_data['stage_1']['std'])
        
        # Plot Stage 0 (solid line, square markers)
        if stage_0_means:
            label = f'{label_prefix} Stage 0' if label_prefix else 'Stage 0'
            ax.plot(stage_0_rounds, stage_0_means, color=color, marker='s', 
                   linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA, linestyle='-')
            
            if show_std and any(std > 0 for std in stage_0_stds):
                stage_0_means_arr = np.array(stage_0_means)
                stage_0_stds_arr = np.array(stage_0_stds)
                ax.fill_between(stage_0_rounds, 
                               stage_0_means_arr - stage_0_stds_arr, 
                               stage_0_means_arr + stage_0_stds_arr,
                               color=color, alpha=STD_ALPHA)
        
        # Plot Stage 1 (dashed line, triangle markers)
        if stage_1_means:
            label = f'{label_prefix} Stage 1' if label_prefix else 'Stage 1'
            ax.plot(stage_1_rounds, stage_1_means, color=color, marker='^', 
                   linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA, linestyle='--')
            
            if show_std and any(std > 0 for std in stage_1_stds):
                stage_1_means_arr = np.array(stage_1_means)
                stage_1_stds_arr = np.array(stage_1_stds)
                ax.fill_between(stage_1_rounds, 
                               stage_1_means_arr - stage_1_stds_arr, 
                               stage_1_means_arr + stage_1_stds_arr,
                               color=color, alpha=STD_ALPHA)
        
        ax.set_xlabel('Federated Round')
    
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def plot_avg_f1_metric(ax, title, server_stats, all_rounds,
                       show_std=True, show_legend=True, color='blue', label_prefix=''):
    """
    Plot Avg_f1 metric (single-series server evaluation across rounds).
    server_stats: dict mapping round -> {'mean': val, 'std': val}
    """
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.12
    
    print("PLOTTING AVG F1 METRIC!!!")
    
    print("Server STATS:", server_stats)
    print("ALL ROUnds:", all_rounds)

    if not server_stats:
        return

    server_rounds = []
    server_means = []
    server_stds = []

    for round_num in sorted(server_stats.keys()):
        if round_num in all_rounds:
            server_rounds.append(round_num)
            server_means.append(server_stats[round_num].get('mean', 0.0))
            server_stds.append(server_stats[round_num].get('std', 0.0))

    if not server_rounds:
        return

    label = f'{label_prefix} (Avg F1)' if label_prefix else 'Avg F1 (Group avg)'
    ax.plot(server_rounds, server_means, color=color, marker='o', linewidth=2, markersize=6,
            label=label, alpha=LINE_ALPHA)

    if show_std and any(std > 0 for std in server_stds):
        server_means_arr = np.array(server_means)
        server_stds_arr = np.array(server_stds)
        ax.fill_between(server_rounds,
                        server_means_arr - server_stds_arr,
                        server_means_arr + server_stds_arr,
                        color=color, alpha=STD_ALPHA)

    ax.set_xlabel('Federated Round')
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def plot_crossfold_group_metric(ax, metric, title, client_stats, server_stats, all_rounds, 
                               global_max_steps_per_round, show_full_training, show_individual_clients, 
                               show_std=True, show_legend=True, color='blue', label_prefix=''):
    """
    Plot a single metric for cross-fold group comparison results
    """
    
    if "test" in metric:
        print(f"Plotting test metric: {metric}")
    
    # Alpha (transparency) settings
    LINE_ALPHA = 0.8
    STD_ALPHA = 0.12  # Reduced for better visibility
    
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
                    ax.set_xticklabels([r for r in plot_rounds])
            
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
                                   smooth_window: int = 0, show_std: bool = True, folds: list = None, name: str = 'crossfold_group_comparison'):
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
    group_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'teal', 'navy', 'maroon']
    
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
            
            # Skip Avg_f1 here - it will be computed after F1/test
            if base_metric == 'Avg_f1':
                continue
            
            if 'test' in base_metric:
                # For test metrics, process both stages separately
                metric_no_stage = f"{base_metric}/{submodel}"
                metric_stage_0 = f"{base_metric}/{submodel}/0"
                metric_stage_1 = f"{base_metric}/{submodel}/1"
                
                client_stats_no_stage, server_stats_no_stage = compute_crossfold_group_statistics(
                    group_data['experiments'], group_data['folds'], metric_no_stage, global_all_rounds,
                    global_max_steps_per_round, show_full_training, show_individual_clients, smooth_window
                )
                
                client_stats_0, server_stats_0 = compute_crossfold_group_statistics(
                    group_data['experiments'], group_data['folds'], metric_stage_0, global_all_rounds, 
                    global_max_steps_per_round, show_full_training, show_individual_clients, smooth_window
                )
                
                client_stats_1, server_stats_1 = compute_crossfold_group_statistics(
                    group_data['experiments'], group_data['folds'], metric_stage_1, global_all_rounds, 
                    global_max_steps_per_round, show_full_training, show_individual_clients, smooth_window
                )
                
                # Combine both stages into a single data structure
                combined_server_stats = {}
                if server_stats_no_stage:
                    for round_num in server_stats_no_stage:
                        if round_num not in combined_server_stats:
                            combined_server_stats[round_num] = {}
                        combined_server_stats[round_num]['stage_0'] = server_stats_no_stage[round_num]
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
                
                # If this is F1 metric, calculate avg_f1 (only available in Stage 2)
                if base_metric == 'F1/test':
                    # Store F1 scores for avg_f1 calculation
                    if group_name not in all_group_stats:
                        all_group_stats[group_name] = {}
                    all_group_stats[group_name]['_f1_stage_0'] = server_stats_0
                    all_group_stats[group_name]['_f1_stage_1'] = server_stats_1
                
                metric = f"{base_metric}/{submodel}"
                all_group_stats[group_name][metric] = (None, combined_server_stats)
            else:
                # For train/val metrics, use the original approach
                metric = f"{base_metric}/{submodel}"
                
                client_stats, server_stats = compute_crossfold_group_statistics(
                    group_data['experiments'], group_data['folds'], metric, global_all_rounds, 
                    global_max_steps_per_round, show_full_training, show_individual_clients, smooth_window
                )
                
                all_group_stats[group_name][metric] = (client_stats, server_stats)
    
    # Compute avg_f1 metric for all groups (after F1/test has been processed)
    for group_name in all_group_stats.keys():
        if '_f1_stage_0' in all_group_stats[group_name] and '_f1_stage_1' in all_group_stats[group_name]:
            f1_stage_0 = all_group_stats[group_name]['_f1_stage_0']
            f1_stage_1 = all_group_stats[group_name]['_f1_stage_1']
            
            # Calculate avg_f1 metric for each round in stage 2
            avg_f1_stats = {}
            all_rounds_sorted = sorted(global_all_rounds)
            
            # Find the midpoint (transition from stage 1 to stage 2)
            midpoint = len(all_rounds_sorted) // 2
            
            # Get the last round of stage 1 for baseline F1 scores
            if midpoint > 0 and midpoint < len(all_rounds_sorted):
                last_stage1_round = all_rounds_sorted[midpoint - 1]
                
                # Get baseline F1 scores at end of stage 1
                f1_s1test_s1_baseline = f1_stage_0.get(last_stage1_round, {}).get('mean', 0)
                f1_s2test_s1_baseline = f1_stage_1.get(last_stage1_round, {}).get('mean', 0)
                
                # Calculate avg_f1 for each round in stage 2
                for round_num in all_rounds_sorted[midpoint:]:
                    # Current F1 scores in stage 2
                    f1_s1test_s2 = f1_stage_0.get(round_num, {}).get('mean', None)
                    f1_s2test_s2 = f1_stage_1.get(round_num, {}).get('mean', None)
                    
                    if f1_s1test_s2 is not None and f1_s2test_s2 is not None:
                        # Calculate avg_f1 metric
                        avg_f1_value = calculate_avg_f1_metric(
                            f1_s1test_s1_baseline,
                            f1_s1test_s2,
                            f1_s2test_s1_baseline,
                            f1_s2test_s2,
                            plasticity_weight=0.5
                        )
                        
                        # Calculate std (simplified - using combined std from both stages)
                        std_s1 = f1_stage_0.get(round_num, {}).get('std', 0)
                        std_s2 = f1_stage_1.get(round_num, {}).get('std', 0)
                        combined_std = np.sqrt(std_s1**2 + std_s2**2) / 2  # Approximate combined std
                        
                        avg_f1_stats[round_num] = {
                            'mean': avg_f1_value,
                            'std': combined_std
                        }
                
                # For stage 1 rounds, set avg_f1 to 0 as it cannot be calculated yet
                for round_num in all_rounds_sorted[:midpoint]:
                    avg_f1_stats[round_num] = {
                        'mean': 0.0,
                        'std': 0.0
                    }
            
            # Store the avg_f1 metric (use same key format as other metrics)
            all_group_stats[group_name][f'Avg_f1/test/{submodel}'] = (None, avg_f1_stats)
    
    print("KEYS:", all_group_stats[group_name].keys())
    
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
            
            # Special-case Avg_f1: plot as a server test-series (no stage split)
            # print("BASE METRIC:", base_metric)
            if 'Avg_f1' in base_metric:
                plot_avg_f1_metric(ax, title, server_stats, global_all_rounds,
                                   show_std=show_std, show_legend=not has_legend_entry, color=color,
                                   label_prefix=group_name)
            # Use specialized function for test metrics to handle stages
            elif 'test' in base_metric:
                plot_crossfold_group_test_metric(ax, metric, title, client_stats, server_stats, global_all_rounds,
                                               global_max_steps_per_round, show_full_training, show_individual_clients,
                                               show_std, show_legend=not has_legend_entry, color=color, 
                                               label_prefix=group_name)
            else:
                plot_crossfold_group_metric(ax, metric, title, client_stats, server_stats, global_all_rounds,
                                           global_max_steps_per_round, show_full_training, show_individual_clients,
                                           show_std, show_legend=not has_legend_entry, color=color, 
                                           label_prefix=group_name)
    
    # Create comparison legend if needed
    if has_legend_entry:
        comparison_legend_handles = []
        
        # Check if we have test metrics to determine legend type
        has_test_metrics = any('test' in base_metric for base_metric in metrics_to_plot.keys() if base_metric != 'Legend')
        
        if has_test_metrics:
            # Create stage-aware legend for test metrics
            for group_idx, group_name in enumerate(all_group_data.keys()):
                color = group_colors[group_idx % len(group_colors)]
                # Stage 0 entry
                line0 = plt.Line2D([0], [0], color=color, linewidth=2, linestyle='-', marker='s',
                                  label=f'{group_name} Stage 0')
                comparison_legend_handles.append(line0)
                # Stage 1 entry
                line1 = plt.Line2D([0], [0], color=color, linewidth=2, linestyle='--', marker='^',
                                  label=f'{group_name} Stage 1')
                comparison_legend_handles.append(line1)
        else:
            # Standard legend for train/val metrics
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
    
    # suffix = "_".join(suffix_parts)
    # group_names_clean = "_vs_".join([name.replace("_", "-") for name in all_group_data.keys()])
    plot_path = os.path.join(ROOT_RESULTS, f"{name}.png")
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
    parser.add_argument('--name', type=str, default='crossfold_group_comparison',
                       help='Base name for the output plot file')
    
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
                                   args.smooth_window, show_std, args.folds, args.name)

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

# python plot_crossfold_group_comparison.py --groups '{"CF Split 1": ["CF_s1", "CF_s2", "CF_s3"], "CF Split 2": ["CF_sp2_s1", "CF_sp2_s2", "CF_sp2_s3"], "CF Split 3": ["CF_sp3_s1", "CF_sp3_s2", "CF_sp3_s3"], "CF Split 4": ["CF_sp4_s1", "CF_sp4_s2", "CF_sp4_s3"], "CF Split 5": ["CF_sp5_s1", "CF_sp5_s2", "CF_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"ID": ["ID_s1", "ID_s2", "ID_s3", "ID_sp2_s1", "ID_sp2_s2", "ID_sp2_s3", "ID_sp3_s1","ID_sp3_s2","ID_sp3_s3","ID_sp4_s1","ID_sp4_s2","ID_sp4_s3","ID_sp5_s1","ID_sp5_s2","ID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"ID Split 1": ["ID_s1", "ID_s2", "ID_s3"], "ID Split 2": ["ID_sp2_s1", "ID_sp2_s2", "ID_sp2_s3"], "ID Split 3": ["ID_sp3_s1","ID_sp3_s2","ID_sp3_s3"], "ID Split 4": ["ID_sp4_s1","ID_sp4_s2","ID_sp4_s3"], "ID Split 5": ["ID_sp5_s1","ID_sp5_s2","ID_sp5_s3"], "CD Split 1": ["CD_s1", "CD_s2", "CD_s3"], "CD Split 2": [ "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3"], "CD Split 3": ["CD_sp3_s1","CD_sp3_s2","CD_sp3_s3"], "CD Split 4": ["CD_sp4_s1","CD_sp4_s2","CD_sp4_s3"], "CD Split 5": ["CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"ID Split 1": ["ID_s1", "ID_s2", "ID_s3"], "ID Split 2": ["ID_sp2_s1", "ID_sp2_s2", "ID_sp2_s3"], "ID Split 3": ["ID_sp3_s1","ID_sp3_s2","ID_sp3_s3"], "CD Split 1": ["CD_s1", "CD_s2", "CD_s3"], "CD Split 2": [ "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3"], "CD Split 3": ["CD_sp3_s1","CD_sp3_s2","CD_sp3_s3"], "CD-Hard Split 1": ["CD_hard_s1", "CD_hard_s2", "CD_hard_s3"], "CD-Hard Split 2": [ "CD_hard_sp2_s1", "CD_hard_sp2_s2", "CD_hard_sp2_s3"], "CD-Hard Split 3": ["CD_hard_sp3_s1","CD_hard_sp3_s2","CD_hard_sp3_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"ID": ["ID_s1", "ID_s2", "ID_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"ID": ["ID_s1", "ID_s2", "ID_s3", "ID_sp2_s1", "ID_sp2_s2", "ID_sp2_s3", "ID_sp3_s1","ID_sp3_s2","ID_sp3_s3","ID_sp4_s1","ID_sp4_s2","ID_sp4_s3","ID_sp5_s1","ID_sp5_s2","ID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"],"CD Hard": ["CD_hard_s1", "CD_hard_s2", "CD_hard_s3", "CD_hard_sp2_s1", "CD_hard_sp2_s2", "CD_hard_sp2_s3", "CD_hard_sp3_s1","CD_hard_sp3_s2","CD_hard_sp3_s3","CD_hard_sp4_s1","CD_hard_sp4_s2","CD_hard_sp4_s3","CD_hard_sp5_s1","CD_hard_sp5_s2","CD_hard_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"ID": ["ID_s1", "ID_s2", "ID_s3", "ID_sp2_s1", "ID_sp2_s2", "ID_sp2_s3", "ID_sp3_s1","ID_sp3_s2","ID_sp3_s3","ID_sp4_s1","ID_sp4_s2","ID_sp4_s3","ID_sp5_s1","ID_sp5_s2","ID_sp5_s3"], "CDCF": ["CDCF_s1", "CDCF_s2", "CDCF_s3", "CDCF_sp2_s1", "CDCF_sp2_s2", "CDCF_sp2_s3", "CDCF_sp3_s1","CDCF_sp3_s2","CDCF_sp3_s3","CDCF_sp4_s1","CDCF_sp4_s2","CDCF_sp4_s3","CDCF_sp5_s1","CDCF_sp5_s2","CDCF_sp5_s3"], "CDCFID": ["CDCFID_s1", "CDCFID_s2", "CDCFID_s3", "CDCFID_sp2_s1", "CDCFID_sp2_s2", "CDCFID_sp2_s3", "CDCFID_sp3_s1","CDCFID_sp3_s2","CDCFID_sp3_s3","CDCFID_sp4_s1","CDCFID_sp4_s2","CDCFID_sp4_s3","CDCFID_sp5_s1","CDCFID_sp5_s2","CDCFID_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDCFID": ["CDCFID_s1", "CDCFID_s2", "CDCFID_s3", "CDCFID_sp2_s1", "CDCFID_sp2_s2", "CDCFID_sp2_s3", "CDCFID_sp3_s1","CDCFID_sp3_s2","CDCFID_sp3_s3","CDCFID_sp4_s1","CDCFID_sp4_s2","CDCFID_sp4_s3","CDCFID_sp5_s1","CDCFID_sp5_s2","CDCFID_sp5_s3"], "CDCF": ["CDCF_s1", "CDCF_s2", "CDCF_s3", "CDCF_sp2_s1", "CDCF_sp2_s2", "CDCF_sp2_s3", "CDCF_sp3_s1","CDCF_sp3_s2","CDCF_sp3_s3","CDCF_sp4_s1","CDCF_sp4_s2","CDCF_sp4_s3","CDCF_sp5_s1","CDCF_sp5_s2","CDCF_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std




#### CF ####
#"ID": ["ID_s1", "ID_s2", "ID_s3", "ID_sp2_s1", "ID_sp2_s2", "ID_sp2_s3", "ID_sp3_s1","ID_sp3_s2","ID_sp3_s3","ID_sp4_s1","ID_sp4_s2","ID_sp4_s3","ID_sp5_s1","ID_sp5_s2","ID_sp5_s3"]


# python plot_crossfold_group_comparison.py --groups '{"CFID": ["ID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1","CFID_sp3_s2","CFID_sp3_s3","CFID_sp4_s1","CFID_sp4_s2","CFID_sp4_s3","CFID_sp5_s1","CFID_sp5_s2","CFID_sp5_s3"], "CF Med 1": ["CF_3_3_430_s1", "CF_3_3_430_s2", "CF_3_3_430_s3", "CF_3_3_430_sp2_s1", "CF_3_3_430_sp2_s2", "CF_3_3_430_sp2_s3", "CF_3_3_430_sp3_s1","CF_3_3_430_sp3_s2","CF_3_3_430_sp3_s3","CF_3_3_430_sp4_s1","CF_3_3_430_sp4_s2","CF_3_3_430_sp4_s3","CF_3_3_430_sp5_s1","CF_3_3_430_sp5_s2","CF_3_3_430_sp5_s3"], "CF Med 2": ["CF_3_3_440_s1", "CF_3_3_440_s2", "CF_3_3_440_s3", "CF_3_3_440_sp2_s1", "CF_3_3_440_sp2_s2", "CF_3_3_440_sp2_s3", "CF_3_3_440_sp3_s1","CF_3_3_440_sp3_s2","CF_3_3_440_sp3_s3","CF_3_3_440_sp4_s1","CF_3_3_440_sp4_s2","CF_3_3_440_sp4_s3","CF_3_3_440_sp5_s1","CF_3_3_440_sp5_s2","CF_3_3_440_sp5_s3"], "CF Hard": ["CF_4_4_450_s1", "CF_4_4_450_s2", "CF_4_4_450_s3", "CF_4_4_450_sp2_s1", "CF_4_4_450_sp2_s2", "CF_4_4_450_sp2_s3", "CF_4_4_450_sp3_s1","CF_4_4_450_sp3_s2","CF_4_4_450_sp3_s3","CF_4_4_450_sp4_s1","CF_4_4_450_sp4_s2","CF_4_4_450_sp4_s3","CF_4_4_450_sp5_s1","CF_4_4_450_sp5_s2","CF_4_4_450_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["ID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1","CFID_sp3_s2","CFID_sp3_s3","CFID_sp4_s1","CFID_sp4_s2","CFID_sp4_s3","CFID_sp5_s1"], "CF Med 1": ["CF_3_3_430_s1", "CF_3_3_430_s2", "CF_3_3_430_s3", "CF_3_3_430_sp2_s1", "CF_3_3_430_sp2_s2", "CF_3_3_430_sp2_s3", "CF_3_3_430_sp3_s1","CF_3_3_430_sp3_s2","CF_3_3_430_sp3_s3","CF_3_3_430_sp4_s1","CF_3_3_430_sp4_s2","CF_3_3_430_sp4_s3","CF_3_3_430_sp5_s1","CF_3_3_430_sp5_s2","CF_3_3_430_sp5_s3"], "CF Med 2": ["CF_3_3_440_s1", "CF_3_3_440_s2", "CF_3_3_440_s3", "CF_3_3_440_sp2_s1", "CF_3_3_440_sp2_s2", "CF_3_3_440_sp2_s3", "CF_3_3_440_sp3_s1","CF_3_3_440_sp3_s2","CF_3_3_440_sp3_s3","CF_3_3_440_sp4_s1","CF_3_3_440_sp4_s2","CF_3_3_440_sp4_s3","CF_3_3_440_sp5_s1"], "CF Hard": ["CF_4_4_450_s1", "CF_4_4_450_s2", "CF_4_4_450_s3", "CF_4_4_450_sp2_s1", "CF_4_4_450_sp2_s2", "CF_4_4_450_sp2_s3", "CF_4_4_450_sp3_s1","CF_4_4_450_sp3_s2","CF_4_4_450_sp3_s3","CF_4_4_450_sp4_s1","CF_4_4_450_sp4_s2","CF_4_4_450_sp4_s3","CF_4_4_450_sp5_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["ID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1","CFID_sp3_s2","CFID_sp3_s3","CFID_sp4_s1","CFID_sp4_s2","CFID_sp4_s3","CFID_sp5_s1"], "CF Med": ["CF_3_3_430_s1", "CF_3_3_430_s2", "CF_3_3_430_s3", "CF_3_3_430_sp2_s1", "CF_3_3_430_sp2_s2", "CF_3_3_430_sp2_s3", "CF_3_3_430_sp3_s1","CF_3_3_430_sp3_s2","CF_3_3_430_sp3_s3","CF_3_3_430_sp4_s1","CF_3_3_430_sp4_s2","CF_3_3_430_sp4_s3","CF_3_3_430_sp5_s1","CF_3_3_430_sp5_s2","CF_3_3_430_sp5_s3"], "CF Hard": ["CF_4_4_450_s1", "CF_4_4_450_s2", "CF_4_4_450_s3", "CF_4_4_450_sp2_s1", "CF_4_4_450_sp2_s2", "CF_4_4_450_sp2_s3", "CF_4_4_450_sp3_s1","CF_4_4_450_sp3_s2","CF_4_4_450_sp3_s3","CF_4_4_450_sp4_s1","CF_4_4_450_sp4_s2","CF_4_4_450_sp4_s3","CF_4_4_450_sp5_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["ID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1"], "CF Med 1": ["CF_3_3_430_s1", "CF_3_3_430_s2", "CF_3_3_430_s3", "CF_3_3_430_sp2_s1", "CF_3_3_430_sp2_s2", "CF_3_3_430_sp2_s3", "CF_3_3_430_sp3_s1"], "CF Hard": ["CF_4_4_450_s1", "CF_4_4_450_s2", "CF_4_4_450_s3", "CF_4_4_450_sp2_s1", "CF_4_4_450_sp2_s2", "CF_4_4_450_sp2_s3", "CF_4_4_450_sp3_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CF_ID_long_s1", "CF_ID_long_s2", "CF_ID_long_s3", "CF_ID_long_sp2_s1", "CF_ID_long_sp2_s2", "CF_ID_long_sp2_s3", "CF_ID_long_sp3_s1","CF_ID_long_sp3_s2","CF_ID_long_sp3_s3","CF_ID_long_sp4_s1","CF_ID_long_sp4_s2","CF_ID_long_sp4_s3","CF_ID_long_sp5_s1","CF_ID_long_sp5_s2","CF_ID_long_sp5_s3"], "CF Blur": ["CF_blur_s1", "CF_blur_s2", "CF_blur_s3", "CF_blur_sp2_s1", "CF_blur_sp2_s2", "CF_blur_sp2_s3", "CF_blur_sp3_s1","CF_blur_sp3_s2","CF_blur_sp3_s3","CF_blur_sp4_s1","CF_blur_sp4_s2","CF_blur_sp4_s3","CF_blur_sp5_s1","CF_blur_sp5_s2","CF_blur_sp5_s3"], "CF Brightness": ["CF_brightness_s1", "CF_brightness_s2", "CF_brightness_s3", "CF_brightness_sp2_s1", "CF_brightness_sp2_s2", "CF_brightness_sp2_s3", "CF_brightness_sp3_s1","CF_brightness_sp3_s2","CF_brightness_sp3_s3","CF_brightness_sp4_s1","CF_brightness_sp4_s2","CF_brightness_sp4_s3","CF_brightness_sp5_s1","CF_brightness_sp5_s2","CF_brightness_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CF_ID_long_s1", "CF_ID_long_s2", "CF_ID_long_s3", "CF_ID_long_sp2_s1", "CF_ID_long_sp2_s2", "CF_ID_long_sp2_s3", "CF_ID_long_sp3_s1","CF_ID_long_sp3_s2","CF_ID_long_sp3_s3","CF_ID_long_sp4_s1","CF_ID_long_sp4_s2","CF_ID_long_sp4_s3","CF_ID_long_sp5_s1","CF_ID_long_sp5_s2","CF_ID_long_sp5_s3"], "CF Blur": ["CF_blur_s1", "CF_blur_s2", "CF_blur_s3", "CF_blur_sp2_s1", "CF_blur_sp2_s2", "CF_blur_sp2_s3", "CF_blur_sp3_s1","CF_blur_sp3_s2","CF_blur_sp3_s3","CF_blur_sp4_s1","CF_blur_sp4_s2","CF_blur_sp4_s3","CF_blur_sp5_s1","CF_blur_sp5_s2","CF_blur_sp5_s3"], "CF Brightness": ["CF_brightness_s1", "CF_brightness_s2", "CF_brightness_s3", "CF_brightness_sp2_s1", "CF_brightness_sp2_s2", "CF_brightness_sp2_s3", "CF_brightness_sp3_s1","CF_brightness_sp3_s2","CF_brightness_sp3_s3","CF_brightness_sp4_s1","CF_brightness_sp4_s2","CF_brightness_sp4_s3","CF_brightness_sp5_s1","CF_brightness_sp5_s2","CF_brightness_sp5_s3"], "CF Brightness nocd": ["CF_brightness_nocd_s1", "CF_brightness_nocd_s2", "CF_brightness_nocd_s3", "CF_brightness_nocd_sp2_s1", "CF_brightness_nocd_sp2_s2", "CF_brightness_nocd_sp2_s3", "CF_brightness_nocd_sp3_s1","CF_brightness_nocd_sp3_s2","CF_brightness_nocd_sp3_s3","CF_brightness_nocd_sp4_s1","CF_brightness_nocd_sp4_s2","CF_brightness_nocd_sp4_s3","CF_brightness_nocd_sp5_s1","CF_brightness_nocd_sp5_s2","CF_brightness_nocd_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CF_ID_long_s1", "CF_ID_long_s2", "CF_ID_long_s3"], "CF Blur": ["CF_blur_s1", "CF_blur_s2", "CF_blur_s3"], "CF Brightness No CD": ["CF_brightness_nocd_s1", "CF_brightness_nocd_s2", "CF_brightness_nocd_s3"], "CF Brightness": ["CF_brightness_s1", "CF_brightness_s2", "CF_brightness_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CF_ID_long_s1", "CF_ID_long_s2", "CF_ID_long_s3", "CF_ID_long_sp2_s1", "CF_ID_long_sp2_s2", "CF_ID_long_sp2_s3", "CF_ID_long_sp3_s1","CF_ID_long_sp3_s2","CF_ID_long_sp3_s3","CF_ID_long_sp4_s1","CF_ID_long_sp4_s2","CF_ID_long_sp4_s3","CF_ID_long_sp5_s1","CF_ID_long_sp5_s2","CF_ID_long_sp5_s3"], "CF Blur": ["CF_blur_s1", "CF_blur_s2", "CF_blur_s3", "CF_blur_sp2_s1", "CF_blur_sp2_s2", "CF_blur_sp2_s3", "CF_blur_sp3_s1","CF_blur_sp3_s2","CF_blur_sp3_s3","CF_blur_sp4_s1","CF_blur_sp4_s2","CF_blur_sp4_s3","CF_blur_sp5_s1","CF_blur_sp5_s2","CF_blur_sp5_s3"], "CF Brightness": ["CF_brightness_s1", "CF_brightness_s2", "CF_brightness_s3", "CF_brightness_sp2_s1", "CF_brightness_sp2_s2", "CF_brightness_sp2_s3", "CF_brightness_sp3_s1","CF_brightness_sp3_s2","CF_brightness_sp3_s3","CF_brightness_sp4_s1","CF_brightness_sp4_s2","CF_brightness_sp4_s3","CF_brightness_sp5_s1","CF_brightness_sp5_s2","CF_brightness_sp5_s3"], "CF Brightness nocd": ["CF_brightness_nocd_s1", "CF_brightness_nocd_s2", "CF_brightness_nocd_s3", "CF_brightness_nocd_sp2_s1", "CF_brightness_nocd_sp2_s2", "CF_brightness_nocd_sp2_s3", "CF_brightness_nocd_sp3_s1","CF_brightness_nocd_sp3_s2","CF_brightness_nocd_sp3_s3","CF_brightness_nocd_sp4_s1","CF_brightness_nocd_sp4_s2","CF_brightness_nocd_sp4_s3","CF_brightness_nocd_sp5_s1","CF_brightness_nocd_sp5_s2","CF_brightness_nocd_sp5_s3"], "CF Brightness Medium": ["CF_brightness_med_s1", "CF_brightness_med_s2", "CF_brightness_med_s3", "CF_brightness_med_sp2_s1", "CF_brightness_med_sp2_s2", "CF_brightness_med_sp2_s3", "CF_brightness_med_sp3_s1","CF_brightness_med_sp3_s2","CF_brightness_med_sp3_s3","CF_brightness_med_sp4_s1","CF_brightness_med_sp4_s2","CF_brightness_med_sp4_s3","CF_brightness_med_sp5_s1","CF_brightness_med_sp5_s2","CF_brightness_med_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std





# python plot_crossfold_group_comparison.py --groups '{"CFID Old": ["CF_ID_long_s1", "CF_ID_long_s2", "CF_ID_long_s3", "CF_ID_long_sp2_s1"], "CF ID 1 EP": ["CF_1ep_s1", "CF_1ep_s2", "CF_1ep_s3", "CF_1ep_sp2_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID Old": ["CF_ID_long_s1", "CF_ID_long_s2", "CF_ID_long_s3", "CF_ID_long_sp2_s1", "CF_ID_long_sp2_s2", "CF_ID_long_sp2_s3", "CF_ID_long_sp3_s1","CF_ID_long_sp3_s2","CF_ID_long_sp3_s3","CF_ID_long_sp4_s1","CF_ID_long_sp4_s2","CF_ID_long_sp4_s3","CF_ID_long_sp5_s1","CF_ID_long_sp5_s2","CF_ID_long_sp5_s3"], "CF ID 1 EP": ["CF_1ep_s1", "CF_1ep_s2", "CF_1ep_s3", "CF_1ep_sp2_s1", "CF_1ep_sp2_s2", "CF_1ep_sp2_s3", "CF_1ep_sp3_s1","CF_1ep_sp3_s2","CF_1ep_sp3_s3","CF_1ep_sp4_s1","CF_1ep_sp4_s2","CF_1ep_sp4_s3","CF_1ep_sp5_s1","CF_1ep_sp5_s2","CF_1ep_sp5_s3"], "CF Brightness": ["CF_brightness_1ep_s1", "CF_brightness_1ep_s2", "CF_brightness_1ep_s3", "CF_brightness_1ep_sp2_s1", "CF_brightness_1ep_sp2_s2", "CF_brightness_1ep_sp2_s3", "CF_brightness_1ep_sp3_s1","CF_brightness_1ep_sp3_s2","CF_brightness_1ep_sp3_s3","CF_brightness_1ep_sp4_s1","CF_brightness_1ep_sp4_s2","CF_brightness_1ep_sp4_s3","CF_brightness_1ep_sp5_s1","CF_brightness_1ep_sp5_s2","CF_brightness_1ep_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CFID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1","CFID_sp3_s2","CFID_sp3_s3","CFID_sp4_s1","CFID_sp4_s2","CFID_sp4_s3","CFID_sp5_s1","CFID_sp5_s2","CFID_sp5_s3"], "CF": ["CF_01_02_07_s1", "CF_01_02_07_s2", "CF_01_02_07_s3", "CF_01_02_07_sp2_s1", "CF_01_02_07_sp2_s2", "CF_01_02_07_sp2_s3", "CF_01_02_07_sp3_s1","CF_01_02_07_sp3_s2","CF_01_02_07_sp3_s3","CF_01_02_07_sp4_s1","CF_01_02_07_sp4_s2","CF_01_02_07_sp4_s3","CF_01_02_07_sp5_s1","CF_01_02_07_sp5_s2","CF_01_02_07_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID Long": ["CFID_Long_s1", "CFID_Long_s2", "CFID_Long_s3", "CFID_Long_sp2_s1", "CFID_Long_sp2_s2", "CFID_Long_sp2_s3", "CFID_Long_sp3_s1","CFID_Long_sp3_s2","CFID_Long_sp3_s3","CFID_Long_sp4_s1","CFID_Long_sp4_s2","CFID_Long_sp4_s3","CFID_Long_sp5_s1","CFID_Long_sp5_s2","CFID_Long_sp5_s3"], "CF Long": ["CF_01_02_07_s1", "CF_01_02_07_Long_s2", "CF_01_02_07_Long_s3", "CF_01_02_07_Long_sp2_s1", "CF_01_02_07_Long_sp2_s2", "CF_01_02_07_Long_sp2_s3", "CF_01_02_07_Long_sp3_s1","CF_01_02_07_Long_sp3_s2","CF_01_02_07_Long_sp3_s3","CF_01_02_07_Long_sp4_s1","CF_01_02_07_Long_sp4_s2","CF_01_02_07_Long_sp4_s3","CF_01_02_07_Long_sp5_s1","CF_01_02_07_Long_sp5_s2","CF_01_02_07_Long_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CFID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1","CFID_sp3_s2","CFID_sp3_s3","CFID_sp4_s1","CFID_sp4_s2","CFID_sp4_s3","CFID_sp5_s1","CFID_sp5_s2","CFID_sp5_s3"], "CF": ["CF_01_02_07_s1", "CF_01_02_07_s2", "CF_01_02_07_s3", "CF_01_02_07_sp2_s1", "CF_01_02_07_sp2_s2", "CF_01_02_07_sp2_s3", "CF_01_02_07_sp3_s1","CF_01_02_07_sp3_s2","CF_01_02_07_sp3_s3","CF_01_02_07_sp4_s1","CF_01_02_07_sp4_s2","CF_01_02_07_sp4_s3","CF_01_02_07_sp5_s1","CF_01_02_07_sp5_s2","CF_01_02_07_sp5_s3"], "CFSampling": ["CF_01_02_07_Sampling_s1", "CF_01_02_07_Sampling_s2", "CF_01_02_07_Sampling_s3", "CF_01_02_07_Sampling_sp2_s1", "CF_01_02_07_Sampling_sp2_s2", "CF_01_02_07_Sampling_sp2_s3", "CF_01_02_07_Sampling_sp3_s1","CF_01_02_07_Sampling_sp3_s2","CF_01_02_07_Sampling_sp3_s3","CF_01_02_07_Sampling_sp4_s1","CF_01_02_07_Sampling_sp4_s2","CF_01_02_07_Sampling_sp4_s3","CF_01_02_07_Sampling_sp5_s1","CF_01_02_07_Sampling_sp5_s2","CF_01_02_07_Sampling_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID": ["CFID_s1", "CFID_s2", "CFID_s3", "CFID_sp2_s1", "CFID_sp2_s2", "CFID_sp2_s3", "CFID_sp3_s1","CFID_sp3_s2","CFID_sp3_s3","CFID_sp4_s1","CFID_sp4_s2","CFID_sp4_s3","CFID_sp5_s1","CFID_sp5_s2","CFID_sp5_s3"], "CF": ["CF_01_02_07_s1", "CF_01_02_07_s2", "CF_01_02_07_s3", "CF_01_02_07_sp2_s1", "CF_01_02_07_sp2_s2", "CF_01_02_07_sp2_s3", "CF_01_02_07_sp3_s1","CF_01_02_07_sp3_s2","CF_01_02_07_sp3_s3","CF_01_02_07_sp4_s1","CF_01_02_07_sp4_s2","CF_01_02_07_sp4_s3","CF_01_02_07_sp5_s1","CF_01_02_07_sp5_s2","CF_01_02_07_sp5_s3"], "CFSampling": ["CF_01_02_07_Sampling_s1", "CF_01_02_07_Sampling_s2", "CF_01_02_07_Sampling_s3", "CF_01_02_07_Sampling_sp2_s1", "CF_01_02_07_Sampling_sp2_s2", "CF_01_02_07_Sampling_sp2_s3", "CF_01_02_07_Sampling_sp3_s1","CF_01_02_07_Sampling_sp3_s2","CF_01_02_07_Sampling_sp3_s3","CF_01_02_07_Sampling_sp4_s1","CF_01_02_07_Sampling_sp4_s2","CF_01_02_07_Sampling_sp4_s3","CF_01_02_07_Sampling_sp5_s1","CF_01_02_07_Sampling_sp5_s2","CF_01_02_07_Sampling_sp5_s3"], "CFSampling2": ["CF_01_02_07_Sampling2_s1", "CF_01_02_07_Sampling2_s2", "CF_01_02_07_Sampling2_s3", "CF_01_02_07_Sampling2_sp2_s1", "CF_01_02_07_Sampling2_sp2_s2", "CF_01_02_07_Sampling2_sp2_s3", "CF_01_02_07_Sampling2_sp3_s1","CF_01_02_07_Sampling2_sp3_s2","CF_01_02_07_Sampling2_sp3_s3","CF_01_02_07_Sampling2_sp4_s1","CF_01_02_07_Sampling2_sp4_s2","CF_01_02_07_Sampling2_sp4_s3","CF_01_02_07_Sampling2_sp5_s1","CF_01_02_07_Sampling2_sp5_s2","CF_01_02_07_Sampling2_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDID": ["CDID_s1", "CDID_s2", "CDID_s3", "CDID_sp2_s1", "CDID_sp2_s2", "CDID_sp2_s3", "CDID_sp3_s1","CDID_sp3_s2","CDID_sp3_s3","CDID_sp4_s1","CDID_sp4_s2","CDID_sp4_s3","CDID_sp5_s1","CDID_sp5_s2","CDID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDID": ["CDID_s1", "CDID_s2", "CDID_s3", "CDID_sp2_s1", "CDID_sp2_s2", "CDID_sp2_s3", "CDID_sp3_s1","CDID_sp3_s2","CDID_sp3_s3","CDID_sp4_s1","CDID_sp4_s2","CDID_sp4_s3","CDID_sp5_s1","CDID_sp5_s2","CDID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"], "CDMethod": ["CDMethod_s1", "CDMethod_s2", "CDMethod_s3", "CDMethod_sp2_s1", "CDMethod_sp2_s2", "CDMethod_sp2_s3", "CDMethod_sp3_s1","CDMethod_sp3_s2","CDMethod_sp3_s3","CDMethod_sp4_s1","CDMethod_sp4_s2","CDMethod_sp4_s3","CDMethod_sp5_s1","CDMethod_sp5_s2","CDMethod_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDID": ["CDID_s1", "CDID_s2", "CDID_s3", "CDID_sp2_s1", "CDID_sp2_s2", "CDID_sp2_s3", "CDID_sp3_s1","CDID_sp3_s2","CDID_sp3_s3","CDID_sp4_s1","CDID_sp4_s2","CDID_sp4_s3","CDID_sp5_s1","CDID_sp5_s2","CDID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"], "CDMethodL": ["CDMethodL_s1", "CDMethodL_s2", "CDMethodL_s3", "CDMethodL_sp2_s1", "CDMethodL_sp2_s2", "CDMethodL_sp2_s3", "CDMethodL_sp3_s1","CDMethodL_sp3_s2","CDMethodL_sp3_s3","CDMethodL_sp4_s1","CDMethodL_sp4_s2","CDMethodL_sp4_s3","CDMethodL_sp5_s1","CDMethodL_sp5_s2","CDMethodL_sp5_s3"], "CDMethodG": ["CDMethodG_s1", "CDMethodG_s2", "CDMethodG_s3", "CDMethodG_sp2_s1", "CDMethodG_sp2_s2", "CDMethodG_sp2_s3", "CDMethodG_sp3_s1","CDMethodG_sp3_s2","CDMethodG_sp3_s3","CDMethodG_sp4_s1","CDMethodG_sp4_s2","CDMethodG_sp4_s3","CDMethodG_sp5_s1","CDMethodG_sp5_s2","CDMethodL_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDID": ["CDID_s1", "CDID_s2", "CDID_s3", "CDID_sp2_s1", "CDID_sp2_s2", "CDID_sp2_s3", "CDID_sp3_s1","CDID_sp3_s2","CDID_sp3_s3","CDID_sp4_s1","CDID_sp4_s2","CDID_sp4_s3","CDID_sp5_s1","CDID_sp5_s2","CDID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"], "CDMethodL": ["CDMethodL_s1", "CDMethodL_s2", "CDMethodL_s3", "CDMethodL_sp2_s1", "CDMethodL_sp2_s2", "CDMethodL_sp2_s3", "CDMethodL_sp3_s1","CDMethodL_sp3_s2","CDMethodL_sp3_s3","CDMethodL_sp4_s1","CDMethodL_sp4_s2","CDMethodL_sp4_s3","CDMethodL_sp5_s1","CDMethodL_sp5_s2","CDMethodL_sp5_s3"], "CDMethodG": ["CDMethodG_s1", "CDMethodG_s2", "CDMethodG_s3", "CDMethodG_sp2_s1", "CDMethodG_sp2_s2", "CDMethodG_sp2_s3", "CDMethodG_sp3_s1","CDMethodG_sp3_s2","CDMethodG_sp3_s3","CDMethodG_sp4_s1","CDMethodG_sp4_s2","CDMethodG_sp4_s3","CDMethodG_sp5_s1","CDMethodG_sp5_s2","CDMethodG_sp5_s3"], "CDMethodG_01t": ["CDMethodG_01t_s1", "CDMethodG_01t_s2", "CDMethodG_01t_s3", "CDMethodG_01t_sp2_s1", "CDMethodG_01t_sp2_s2", "CDMethodG_01t_sp2_s3", "CDMethodG_01t_sp3_s1","CDMethodG_01t_sp3_s2","CDMethodG_01t_sp3_s3","CDMethodG_01t_sp4_s1","CDMethodG_01t_sp4_s2","CDMethodG_01t_sp4_s3","CDMethodG_01t_sp5_s1","CDMethodG_01t_sp5_s2","CDMethodG_01t_sp5_s3"], "CD_swap_21_MG_no_var": ["CD_swap_21_MG_no_var_s1", "CD_swap_21_MG_no_var_s2", "CD_swap_21_MG_no_var_s3", "CD_swap_21_MG_no_var_sp2_s1", "CD_swap_21_MG_no_var_sp2_s2", "CD_swap_21_MG_no_var_sp2_s3", "CD_swap_21_MG_no_var_sp3_s1","CD_swap_21_MG_no_var_sp3_s2","CD_swap_21_MG_no_var_sp3_s3","CD_swap_21_MG_no_var_sp4_s1","CD_swap_21_MG_no_var_sp4_s2","CD_swap_21_MG_no_var_sp4_s3","CD_swap_21_MG_no_var_sp5_s1","CD_swap_21_MG_no_var_sp5_s2","CD_swap_21_MG_no_var_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDID": ["CDID_s1", "CDID_s2", "CDID_s3", "CDID_sp2_s1", "CDID_sp2_s2", "CDID_sp2_s3", "CDID_sp3_s1","CDID_sp3_s2","CDID_sp3_s3","CDID_sp4_s1","CDID_sp4_s2","CDID_sp4_s3","CDID_sp5_s1","CDID_sp5_s2","CDID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"], "CD_swap_21": ["CD_swap_21_s1", "CD_swap_21_s2", "CD_swap_21_s3", "CD_swap_21_sp2_s1", "CD_swap_21_sp2_s2", "CD_swap_21_sp2_s3", "CD_swap_21_sp3_s1","CD_swap_21_sp3_s2","CD_swap_21_sp3_s3","CD_swap_21_sp4_s1","CD_swap_21_sp4_s2","CD_swap_21_sp4_s3","CD_swap_21_sp5_s1","CD_swap_21_sp5_s2","CD_swap_21_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CDID": ["CDID_s1", "CDID_s2", "CDID_s3", "CDID_sp2_s1", "CDID_sp2_s2", "CDID_sp2_s3", "CDID_sp3_s1","CDID_sp3_s2","CDID_sp3_s3","CDID_sp4_s1","CDID_sp4_s2","CDID_sp4_s3","CDID_sp5_s1","CDID_sp5_s2","CDID_sp5_s3"], "CD": ["CD_s1", "CD_s2", "CD_s3", "CD_sp2_s1", "CD_sp2_s2", "CD_sp2_s3", "CD_sp3_s1","CD_sp3_s2","CD_sp3_s3","CD_sp4_s1","CD_sp4_s2","CD_sp4_s3","CD_sp5_s1","CD_sp5_s2","CD_sp5_s3"], "CD_swap_21": ["CD_swap_21_s1", "CD_swap_21_s2", "CD_swap_21_s3", "CD_swap_21_sp2_s1", "CD_swap_21_sp2_s2", "CD_swap_21_sp2_s3", "CD_swap_21_sp3_s1","CD_swap_21_sp3_s2","CD_swap_21_sp3_s3","CD_swap_21_sp4_s1","CD_swap_21_sp4_s2","CD_swap_21_sp4_s3","CD_swap_21_sp5_s1","CD_swap_21_sp5_s2","CD_swap_21_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID 1EP": ["CFID_1ep_s1", "CFID_1ep_s2", "CFID_1ep_s3", "CFID_1ep_sp2_s1", "CFID_1ep_sp2_s2", "CFID_1ep_sp2_s3", "CFID_1ep_sp3_s1","CFID_1ep_sp3_s2","CFID_1ep_sp3_s3","CFID_1ep_sp4_s1","CFID_1ep_sp4_s2","CFID_1ep_sp4_s3","CFID_1ep_sp5_s1","CFID_1ep_sp5_s2","CFID_1ep_sp5_s3"], "CF 1EP": ["CF_01_02_07_1ep_s1", "CF_01_02_07_1ep_s2", "CF_01_02_07_1ep_s3", "CF_01_02_07_1ep_sp2_s1", "CF_01_02_07_1ep_sp2_s2", "CF_01_02_07_1ep_sp2_s3", "CF_01_02_07_1ep_sp3_s1","CF_01_02_07_1ep_sp3_s2","CF_01_02_07_1ep_sp3_s3","CF_01_02_07_1ep_sp4_s1","CF_01_02_07_1ep_sp4_s2","CF_01_02_07_1ep_sp4_s3","CF_01_02_07_1ep_sp5_s1","CF_01_02_07_1ep_sp5_s2","CF_01_02_07_1ep_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"], "CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"], "CFCD Med": ["CFCD_med_s1", "CFCD_med_s2", "CFCD_med_s3", "CFCD_med_sp2_s1", "CFCD_med_sp2_s2", "CFCD_med_sp2_s3", "CFCD_med_sp3_s1","CFCD_med_sp3_s2","CFCD_med_sp3_s3","CFCD_med_sp4_s1","CFCD_med_sp4_s2","CFCD_med_sp4_s3","CFCD_med_sp5_s1","CFCD_med_sp5_s2","CFCD_med_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"], "CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"], "CFCD Med": ["CFCD_med_s1", "CFCD_med_s2", "CFCD_med_s3", "CFCD_med_sp2_s1", "CFCD_med_sp2_s2", "CFCD_med_sp2_s3", "CFCD_med_sp3_s1","CFCD_med_sp3_s2","CFCD_med_sp3_s3","CFCD_med_sp4_s1","CFCD_med_sp4_s2","CFCD_med_sp4_s3","CFCD_med_sp5_s1","CFCD_med_sp5_s2","CFCD_med_sp5_s3"], "CFCD Med2": ["CFCD_med2_s1", "CFCD_med2_s2", "CFCD_med2_s3", "CFCD_med2_sp2_s1", "CFCD_med2_sp2_s2", "CFCD_med2_sp2_s3", "CFCD_med2_sp3_s1","CFCD_med2_sp3_s2","CFCD_med2_sp3_s3","CFCD_med2_sp4_s1","CFCD_med2_sp4_s2","CFCD_med2_sp4_s3","CFCD_med2_sp5_s1","CFCD_med2_sp5_s2","CFCD_med_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID Long": ["CFID_Long_s1", "CFID_Long_s2", "CFID_Long_s3", "CFID_Long_sp2_s1", "CFID_Long_sp2_s2", "CFID_Long_sp2_s3", "CFID_Long_sp3_s1","CFID_Long_sp3_s2","CFID_Long_sp3_s3","CFID_Long_sp4_s1","CFID_Long_sp4_s2","CFID_Long_sp4_s3","CFID_Long_sp5_s1","CFID_Long_sp5_s2","CFID_Long_sp5_s3"], "CF Long": ["CF_01_02_07_s1", "CF_01_02_07_Long_s2", "CF_01_02_07_Long_s3", "CF_01_02_07_Long_sp2_s1", "CF_01_02_07_Long_sp2_s2", "CF_01_02_07_Long_sp2_s3", "CF_01_02_07_Long_sp3_s1","CF_01_02_07_Long_sp3_s2","CF_01_02_07_Long_sp3_s3","CF_01_02_07_Long_sp4_s1","CF_01_02_07_Long_sp4_s2","CF_01_02_07_Long_sp4_s3","CF_01_02_07_Long_sp5_s1","CF_01_02_07_Long_sp5_s2","CF_01_02_07_Long_sp5_s3"], "CFID NO CD": ["CFID_nocd_s1", "CFID_nocd_s2", "CFID_nocd_s3", "CFID_nocd_sp2_s1", "CFID_nocd_sp2_s2", "CFID_nocd_sp2_s3", "CFID_nocd_sp3_s1","CFID_nocd_sp3_s2","CFID_nocd_sp3_s3","CFID_nocd_sp4_s1","CFID_nocd_sp4_s2","CFID_nocd_sp4_s3","CFID_nocd_sp5_s1","CFID_nocd_sp5_s2","CFID_nocd_sp5_s3"], "CF NO CD": ["CF_nocd_s1", "CF_nocd_s2", "CF_nocd_s3", "CF_nocd_sp2_s1", "CF_nocd_sp2_s2", "CF_nocd_sp2_s3", "CF_nocd_sp3_s1","CF_nocd_sp3_s2","CF_nocd_sp3_s3","CF_nocd_sp4_s1","CF_nocd_sp4_s2","CF_nocd_sp4_s3","CF_nocd_sp5_s1","CF_nocd_sp5_s2","CF_nocd_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"], "CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"],"CFCDIDnew": ["CFCDIDnew_s1", "CFCDIDnew_s2", "CFCDIDnew_s3", "CFCDIDnew_sp2_s1", "CFCDIDnew_sp2_s2", "CFCDIDnew_sp2_s3", "CFCDIDnew_sp3_s1","CFCDIDnew_sp3_s2","CFCDIDnew_sp3_s3","CFCDIDnew_sp4_s1","CFCDIDnew_sp4_s2","CFCDIDnew_sp4_s3","CFCDIDnew_sp5_s1","CFCDIDnew_sp5_s2","CFCDIDnew_sp5_s3"], "CFCDnew_swap": ["CFCDnew_swap_s1", "CFCDnew_swap_s2", "CFCDnew_swap_s3", "CFCDnew_swap_sp2_s1", "CFCDnew_swap_sp2_s2", "CFCDnew_swap_sp2_s3", "CFCDnew_swap_sp3_s1","CFCDnew_swap_sp3_s2","CFCDnew_swap_sp3_s3","CFCDnew_swap_sp4_s1","CFCDnew_swap_sp4_s2","CFCDnew_swap_sp4_s3","CFCDnew_swap_sp5_s1","CFCDnew_swap_sp5_s2","CFCDnew_swap_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID Long": ["CFID_Long_s1", "CFID_Long_s2", "CFID_Long_s3", "CFID_Long_sp2_s1", "CFID_Long_sp2_s2", "CFID_Long_sp2_s3", "CFID_Long_sp3_s1","CFID_Long_sp3_s2","CFID_Long_sp3_s3","CFID_Long_sp4_s1","CFID_Long_sp4_s2","CFID_Long_sp4_s3","CFID_Long_sp5_s1","CFID_Long_sp5_s2","CFID_Long_sp5_s3"], "CF Long": ["CF_01_02_07_s1", "CF_01_02_07_Long_s2", "CF_01_02_07_Long_s3", "CF_01_02_07_Long_sp2_s1", "CF_01_02_07_Long_sp2_s2", "CF_01_02_07_Long_sp2_s3", "CF_01_02_07_Long_sp3_s1","CF_01_02_07_Long_sp3_s2","CF_01_02_07_Long_sp3_s3","CF_01_02_07_Long_sp4_s1","CF_01_02_07_Long_sp4_s2","CF_01_02_07_Long_sp4_s3","CF_01_02_07_Long_sp5_s1","CF_01_02_07_Long_sp5_s2","CF_01_02_07_Long_sp5_s3"], "CFID NO CD": ["CFID_nocd_s1", "CFID_nocd_s2", "CFID_nocd_s3", "CFID_nocd_sp2_s1", "CFID_nocd_sp2_s2", "CFID_nocd_sp2_s3", "CFID_nocd_sp3_s1","CFID_nocd_sp3_s2","CFID_nocd_sp3_s3","CFID_nocd_sp4_s1","CFID_nocd_sp4_s2","CFID_nocd_sp4_s3","CFID_nocd_sp5_s1","CFID_nocd_sp5_s2","CFID_nocd_sp5_s3"], "CF NO CD": ["CF_nocd_s1", "CF_nocd_s2", "CF_nocd_s3", "CF_nocd_sp2_s1", "CF_nocd_sp2_s2", "CF_nocd_sp2_s3", "CF_nocd_sp3_s1","CF_nocd_sp3_s2","CF_nocd_sp3_s3","CF_nocd_sp4_s1","CF_nocd_sp4_s2","CF_nocd_sp4_s3","CF_nocd_sp5_s1","CF_nocd_sp5_s2","CF_nocd_sp5_s3"], "CF balanced Test": ["CF_nocd_1_each_s1", "CF_nocd_1_each_s2", "CF_nocd_1_each_s3", "CF_nocd_1_each_sp2_s1", "CF_nocd_1_each_sp2_s2", "CF_nocd_1_each_sp2_s3", "CF_nocd_1_each_sp3_s1","CF_nocd_1_each_sp3_s2","CF_nocd_1_each_sp3_s3","CF_nocd_1_each_sp4_s1","CF_nocd_1_each_sp4_s2","CF_nocd_1_each_sp4_s3","CF_nocd_1_each_sp5_s1","CF_nocd_1_each_sp5_s2","CF_nocd_1_each_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFID Long": ["CFID_Long_s1", "CFID_Long_s2", "CFID_Long_s3", "CFID_Long_sp2_s1", "CFID_Long_sp2_s2", "CFID_Long_sp2_s3", "CFID_Long_sp3_s1","CFID_Long_sp3_s2","CFID_Long_sp3_s3","CFID_Long_sp4_s1","CFID_Long_sp4_s2","CFID_Long_sp4_s3","CFID_Long_sp5_s1","CFID_Long_sp5_s2","CFID_Long_sp5_s3"], "CFID NO CD": ["CFID_nocd_s1", "CFID_nocd_s2", "CFID_nocd_s3", "CFID_nocd_sp2_s1", "CFID_nocd_sp2_s2", "CFID_nocd_sp2_s3", "CFID_nocd_sp3_s1","CFID_nocd_sp3_s2","CFID_nocd_sp3_s3","CFID_nocd_sp4_s1","CFID_nocd_sp4_s2","CFID_nocd_sp4_s3","CFID_nocd_sp5_s1","CFID_nocd_sp5_s2","CFID_nocd_sp5_s3"], "CF NO CD": ["CF_nocd_s1", "CF_nocd_s2", "CF_nocd_s3", "CF_nocd_sp2_s1", "CF_nocd_sp2_s2", "CF_nocd_sp2_s3", "CF_nocd_sp3_s1","CF_nocd_sp3_s2","CF_nocd_sp3_s3","CF_nocd_sp4_s1","CF_nocd_sp4_s2","CF_nocd_sp4_s3","CF_nocd_sp5_s1","CF_nocd_sp5_s2","CF_nocd_sp5_s3"], "CF balanced Test": ["CF_nocd_1_each_s1", "CF_nocd_1_each_s2", "CF_nocd_1_each_s3", "CF_nocd_1_each_sp2_s1", "CF_nocd_1_each_sp2_s2", "CF_nocd_1_each_sp2_s3", "CF_nocd_1_each_sp3_s1","CF_nocd_1_each_sp3_s2","CF_nocd_1_each_sp3_s3","CF_nocd_1_each_sp4_s1","CF_nocd_1_each_sp4_s2","CF_nocd_1_each_sp4_s3","CF_nocd_1_each_sp5_s1","CF_nocd_1_each_sp5_s2","CF_nocd_1_each_sp5_s3"], "CFID balanced Test": ["CFID_nocd_1_each_s1", "CFID_nocd_1_each_s2", "CFID_nocd_1_each_s3", "CFID_nocd_1_each_sp2_s1", "CFID_nocd_1_each_sp2_s2", "CFID_nocd_1_each_sp2_s3", "CFID_nocd_1_each_sp3_s1","CFID_nocd_1_each_sp3_s2","CFID_nocd_1_each_sp3_s3","CFID_nocd_1_each_sp4_s1","CFID_nocd_1_each_sp4_s2","CFID_nocd_1_each_sp4_s3","CFID_nocd_1_each_sp5_s1","CFID_nocd_1_each_sp5_s2","CFID_nocd_1_each_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"], "CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"],"CFCDIDnew": ["CFCDIDnew_s1", "CFCDIDnew_s2", "CFCDIDnew_s3", "CFCDIDnew_sp2_s1", "CFCDIDnew_sp2_s2", "CFCDIDnew_sp2_s3", "CFCDIDnew_sp3_s1","CFCDIDnew_sp3_s2","CFCDIDnew_sp3_s3","CFCDIDnew_sp4_s1","CFCDIDnew_sp4_s2","CFCDIDnew_sp4_s3","CFCDIDnew_sp5_s1","CFCDIDnew_sp5_s2","CFCDIDnew_sp5_s3"], "CFCDnew_swap": ["CFCDnew_swap_s1", "CFCDnew_swap_s2", "CFCDnew_swap_s3", "CFCDnew_swap_sp2_s1", "CFCDnew_swap_sp2_s2", "CFCDnew_swap_sp2_s3", "CFCDnew_swap_sp3_s1","CFCDnew_swap_sp3_s2","CFCDnew_swap_sp3_s3","CFCDnew_swap_sp4_s1","CFCDnew_swap_sp4_s2","CFCDnew_swap_sp4_s3","CFCDnew_swap_sp5_s1","CFCDnew_swap_sp5_s2","CFCDnew_swap_sp5_s3"],"CFCD Balanced Test": ["CFCDnew_swap_1_each_s1", "CFCDnew_swap_1_each_s2", "CFCDnew_swap_1_each_s3", "CFCDnew_swap_1_each_sp2_s1", "CFCDnew_swap_1_each_sp2_s2", "CFCDnew_swap_1_each_sp2_s3", "CFCDnew_swap_1_each_sp3_s1","CFCDnew_swap_1_each_sp3_s2","CFCDnew_swap_1_each_sp3_s3","CFCDnew_swap_1_each_sp4_s1","CFCDnew_swap_1_each_sp4_s2","CFCDnew_swap_1_each_sp4_s3","CFCDnew_swap_1_each_sp5_s1","CFCDnew_swap_1_each_sp5_s2","CFCDnew_swap_1_each_sp5_s3"], "CFCDID Balanced": ["CFCDIDnew_1_each_s1", "CFCDIDnew_1_each_s2", "CFCDIDnew_1_each_s3", "CFCDIDnew_1_each_sp2_s1", "CFCDIDnew_1_each_sp2_s2", "CFCDIDnew_1_each_sp2_s3", "CFCDIDnew_1_each_sp3_s1","CFCDIDnew_1_each_sp3_s2","CFCDIDnew_1_each_sp3_s3","CFCDIDnew_1_each_sp4_s1","CFCDIDnew_1_each_sp4_s2","CFCDIDnew_1_each_sp4_s3","CFCDIDnew_1_each_sp5_s1","CFCDIDnew_1_each_sp5_s2","CFCDIDnew_1_each_sp5_s3"],"CFCD Full": ["CFCDnew_swap_1_each_fullaug_s1", "CFCDnew_swap_1_each_fullaug_s2", "CFCDnew_swap_1_each_fullaug_s3", "CFCDnew_swap_1_each_fullaug_sp2_s1", "CFCDnew_swap_1_each_fullaug_sp2_s2", "CFCDnew_swap_1_each_fullaug_sp2_s3", "CFCDnew_swap_1_each_fullaug_sp3_s1","CFCDnew_swap_1_each_fullaug_sp3_s2","CFCDnew_swap_1_each_fullaug_sp3_s3","CFCDnew_swap_1_each_fullaug_sp4_s1","CFCDnew_swap_1_each_fullaug_sp4_s2","CFCDnew_swap_1_each_fullaug_sp4_s3","CFCDnew_swap_1_each_fullaug_sp5_s1","CFCDnew_swap_1_each_fullaug_sp5_s2","CFCDnew_swap_1_each_fullaug_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CF_nocd_1_each_same_test": ["CF_nocd_1_each_same_test_s1", "CF_nocd_1_each_same_test_s2", "CF_nocd_1_each_same_test_s3", "CF_nocd_1_each_same_test_sp2_s1", "CF_nocd_1_each_same_test_sp2_s2", "CF_nocd_1_each_same_test_sp2_s3", "CF_nocd_1_each_same_test_sp3_s1","CF_nocd_1_each_same_test_sp3_s2","CF_nocd_1_each_same_test_sp3_s3","CF_nocd_1_each_same_test_sp4_s1","CF_nocd_1_each_same_test_sp4_s2","CF_nocd_1_each_same_test_sp4_s3","CF_nocd_1_each_same_test_sp5_s1","CF_nocd_1_each_same_test_sp5_s2","CF_nocd_1_each_same_test_sp5_s3"], "CFID_nocd_same_1_each": ["CFID_nocd_same_1_each_s1", "CFID_nocd_same_1_each_s2", "CFID_nocd_same_1_each_s3", "CFID_nocd_same_1_each_sp2_s1", "CFID_nocd_same_1_each_sp2_s2", "CFID_nocd_same_1_each_sp2_s3", "CFID_nocd_same_1_each_sp3_s1","CFID_nocd_same_1_each_sp3_s2","CFID_nocd_same_1_each_sp3_s3","CFID_nocd_same_1_each_sp4_s1","CFID_nocd_same_1_each_sp4_s2","CFID_nocd_same_1_each_sp4_s3","CFID_nocd_same_1_each_sp5_s1","CFID_nocd_same_1_each_sp5_s2","CFID_nocd_same_1_each_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFCDIDnew_same_1_each": ["CFCDIDnew_same_1_each_s1", "CFCDIDnew_same_1_each_s2", "CFCDIDnew_same_1_each_s3", "CFCDIDnew_same_1_each_sp2_s1", "CFCDIDnew_same_1_each_sp2_s2", "CFCDIDnew_same_1_each_sp2_s3", "CFCDIDnew_same_1_each_sp3_s1","CFCDIDnew_same_1_each_sp3_s2","CFCDIDnew_same_1_each_sp3_s3","CFCDIDnew_same_1_each_sp4_s1","CFCDIDnew_same_1_each_sp4_s2","CFCDIDnew_same_1_each_sp4_s3","CFCDIDnew_same_1_each_sp5_s1","CFCDIDnew_same_1_each_sp5_s2","CFCDIDnew_same_1_each_sp5_s3"], "CFCDnew_swap_same_1_each_fullaug": ["CFCDnew_swap_same_1_each_fullaug_s1", "CFCDnew_swap_same_1_each_fullaug_s2", "CFCDnew_swap_same_1_each_fullaug_s3", "CFCDnew_swap_same_1_each_fullaug_sp2_s1", "CFCDnew_swap_same_1_each_fullaug_sp2_s2", "CFCDnew_swap_same_1_each_fullaug_sp2_s3", "CFCDnew_swap_same_1_each_fullaug_sp3_s1","CFCDnew_swap_same_1_each_fullaug_sp3_s2","CFCDnew_swap_same_1_each_fullaug_sp3_s3","CFCDnew_swap_same_1_each_fullaug_sp4_s1","CFCDnew_swap_same_1_each_fullaug_sp4_s2","CFCDnew_swap_same_1_each_fullaug_sp4_s3","CFCDnew_swap_same_1_each_fullaug_sp5_s1","CFCDnew_swap_same_1_each_fullaug_sp5_s2","CFCDnew_swap_same_1_each_fullaug_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std

# python plot_crossfold_group_comparison.py --groups '{"CFCDID Balanced": ["CFCDIDnew_1_each_s1", "CFCDIDnew_1_each_s2", "CFCDIDnew_1_each_s3", "CFCDIDnew_1_each_sp2_s1", "CFCDIDnew_1_each_sp2_s2", "CFCDIDnew_1_each_sp2_s3", "CFCDIDnew_1_each_sp3_s1","CFCDIDnew_1_each_sp3_s2","CFCDIDnew_1_each_sp3_s3","CFCDIDnew_1_each_sp4_s1","CFCDIDnew_1_each_sp4_s2","CFCDIDnew_1_each_sp4_s3","CFCDIDnew_1_each_sp5_s1","CFCDIDnew_1_each_sp5_s2","CFCDIDnew_1_each_sp5_s3"],"CFCD Full": ["CFCDnew_swap_1_each_fullaug_s1", "CFCDnew_swap_1_each_fullaug_s2", "CFCDnew_swap_1_each_fullaug_s3", "CFCDnew_swap_1_each_fullaug_sp2_s1", "CFCDnew_swap_1_each_fullaug_sp2_s2", "CFCDnew_swap_1_each_fullaug_sp2_s3", "CFCDnew_swap_1_each_fullaug_sp3_s1","CFCDnew_swap_1_each_fullaug_sp3_s2","CFCDnew_swap_1_each_fullaug_sp3_s3","CFCDnew_swap_1_each_fullaug_sp4_s1","CFCDnew_swap_1_each_fullaug_sp4_s2","CFCDnew_swap_1_each_fullaug_sp4_s3","CFCDnew_swap_1_each_fullaug_sp5_s1","CFCDnew_swap_1_each_fullaug_sp5_s2","CFCDnew_swap_1_each_fullaug_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std








# python plot_crossfold_group_comparison.py --groups '{"CFCDIDnew_same_1_each": ["CFCDIDnew_same_1_each_s1", "CFCDIDnew_same_1_each_s2", "CFCDIDnew_same_1_each_s3", "CFCDIDnew_same_1_each_sp2_s1", "CFCDIDnew_same_1_each_sp2_s2", "CFCDIDnew_same_1_each_sp2_s3", "CFCDIDnew_same_1_each_sp3_s1","CFCDIDnew_same_1_each_sp3_s2","CFCDIDnew_same_1_each_sp3_s3","CFCDIDnew_same_1_each_sp4_s1","CFCDIDnew_same_1_each_sp4_s2","CFCDIDnew_same_1_each_sp4_s3","CFCDIDnew_same_1_each_sp5_s1","CFCDIDnew_same_1_each_sp5_s2","CFCDIDnew_same_1_each_sp5_s3"], "CFCDnew_swap_same_1_each_fullaug": ["CFCDnew_swap_same_1_each_fullaug_s1", "CFCDnew_swap_same_1_each_fullaug_s2", "CFCDnew_swap_same_1_each_fullaug_s3", "CFCDnew_swap_same_1_each_fullaug_sp2_s1", "CFCDnew_swap_same_1_each_fullaug_sp2_s2", "CFCDnew_swap_same_1_each_fullaug_sp2_s3", "CFCDnew_swap_same_1_each_fullaug_sp3_s1","CFCDnew_swap_same_1_each_fullaug_sp3_s2","CFCDnew_swap_same_1_each_fullaug_sp3_s3","CFCDnew_swap_same_1_each_fullaug_sp4_s1","CFCDnew_swap_same_1_each_fullaug_sp4_s2","CFCDnew_swap_same_1_each_fullaug_sp4_s3","CFCDnew_swap_same_1_each_fullaug_sp5_s1","CFCDnew_swap_same_1_each_fullaug_sp5_s2","CFCDnew_swap_same_1_each_fullaug_sp5_s3"], "MG 02": ["MG_02_s1", "MG_02_s2", "MG_02_s3", "MG_02_sp2_s1", "MG_02_sp2_s2", "MG_02_sp2_s3", "MG_02_sp3_s1","MG_02_sp3_s2","MG_02_sp3_s3","MG_02_sp4_s1","MG_02_sp4_s2","MG_02_sp4_s3","MG_02_sp5_s1","MG_02_sp5_s2","MG_02_sp5_s3"], "MG 05": ["MG_05_s1", "MG_05_s2", "MG_05_s3", "MG_05_sp2_s1", "MG_05_sp2_s2", "MG_05_sp2_s3", "MG_05_sp3_s1","MG_05_sp3_s2","MG_05_sp3_s3","MG_05_sp4_s1","MG_05_sp4_s2","MG_05_sp4_s3","MG_05_sp5_s1","MG_05_sp5_s2","MG_05_sp5_s3"], "MG 08": ["MG_08_s1", "MG_08_s2", "MG_08_s3", "MG_08_sp2_s1", "MG_08_sp2_s2", "MG_08_sp2_s3", "MG_08_sp3_s1","MG_08_sp3_s2","MG_08_sp3_s3","MG_08_sp4_s1","MG_08_sp4_s2","MG_08_sp4_s3","MG_08_sp5_s1","MG_08_sp5_s2","MG_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name "MG"


# python plot_crossfold_group_comparison.py --groups '{"CFCDIDnew_same_1_each": ["CFCDIDnew_same_1_each_s1", "CFCDIDnew_same_1_each_s2", "CFCDIDnew_same_1_each_s3", "CFCDIDnew_same_1_each_sp2_s1", "CFCDIDnew_same_1_each_sp2_s2", "CFCDIDnew_same_1_each_sp2_s3", "CFCDIDnew_same_1_each_sp3_s1","CFCDIDnew_same_1_each_sp3_s2","CFCDIDnew_same_1_each_sp3_s3","CFCDIDnew_same_1_each_sp4_s1","CFCDIDnew_same_1_each_sp4_s2","CFCDIDnew_same_1_each_sp4_s3","CFCDIDnew_same_1_each_sp5_s1","CFCDIDnew_same_1_each_sp5_s2","CFCDIDnew_same_1_each_sp5_s3"], "CFCDnew_swap_same_1_each_fullaug": ["CFCDnew_swap_same_1_each_fullaug_s1", "CFCDnew_swap_same_1_each_fullaug_s2", "CFCDnew_swap_same_1_each_fullaug_s3", "CFCDnew_swap_same_1_each_fullaug_sp2_s1", "CFCDnew_swap_same_1_each_fullaug_sp2_s2", "CFCDnew_swap_same_1_each_fullaug_sp2_s3", "CFCDnew_swap_same_1_each_fullaug_sp3_s1","CFCDnew_swap_same_1_each_fullaug_sp3_s2","CFCDnew_swap_same_1_each_fullaug_sp3_s3","CFCDnew_swap_same_1_each_fullaug_sp4_s1","CFCDnew_swap_same_1_each_fullaug_sp4_s2","CFCDnew_swap_same_1_each_fullaug_sp4_s3","CFCDnew_swap_same_1_each_fullaug_sp5_s1","CFCDnew_swap_same_1_each_fullaug_sp5_s2","CFCDnew_swap_same_1_each_fullaug_sp5_s3"], "ML 02": ["ML_02_s1", "ML_02_s2", "ML_02_s3", "ML_02_sp2_s1", "ML_02_sp2_s2", "ML_02_sp2_s3", "ML_02_sp3_s1","ML_02_sp3_s2","ML_02_sp3_s3","ML_02_sp4_s1","ML_02_sp4_s2","ML_02_sp4_s3","ML_02_sp5_s1","ML_02_sp5_s2","ML_02_sp5_s3"], "ML 05": ["ML_05_s1", "ML_05_s2", "ML_05_s3", "ML_05_sp2_s1", "ML_05_sp2_s2", "ML_05_sp2_s3", "ML_05_sp3_s1","ML_05_sp3_s2","ML_05_sp3_s3","ML_05_sp4_s1","ML_05_sp4_s2","ML_05_sp4_s3","ML_05_sp5_s1","ML_05_sp5_s2","ML_05_sp5_s3"], "ML 08": ["ML_08_s1", "ML_08_s2", "ML_08_s3", "ML_08_sp2_s1", "ML_08_sp2_s2", "ML_08_sp2_s3", "ML_08_sp3_s1","ML_08_sp3_s2","ML_08_sp3_s3","ML_08_sp4_s1","ML_08_sp4_s2","ML_08_sp4_s3","ML_08_sp5_s1","ML_08_sp5_s2","ML_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name "ML"


# python plot_crossfold_group_comparison.py --groups '{"CFCDIDnew_same_1_each": ["CFCDIDnew_same_1_each_s1", "CFCDIDnew_same_1_each_s2", "CFCDIDnew_same_1_each_s3", "CFCDIDnew_same_1_each_sp2_s1", "CFCDIDnew_same_1_each_sp2_s2", "CFCDIDnew_same_1_each_sp2_s3", "CFCDIDnew_same_1_each_sp3_s1","CFCDIDnew_same_1_each_sp3_s2","CFCDIDnew_same_1_each_sp3_s3","CFCDIDnew_same_1_each_sp4_s1","CFCDIDnew_same_1_each_sp4_s2","CFCDIDnew_same_1_each_sp4_s3","CFCDIDnew_same_1_each_sp5_s1","CFCDIDnew_same_1_each_sp5_s2","CFCDIDnew_same_1_each_sp5_s3"], "CFCDnew_swap_same_1_each_fullaug": ["CFCDnew_swap_same_1_each_fullaug_s1", "CFCDnew_swap_same_1_each_fullaug_s2", "CFCDnew_swap_same_1_each_fullaug_s3", "CFCDnew_swap_same_1_each_fullaug_sp2_s1", "CFCDnew_swap_same_1_each_fullaug_sp2_s2", "CFCDnew_swap_same_1_each_fullaug_sp2_s3", "CFCDnew_swap_same_1_each_fullaug_sp3_s1","CFCDnew_swap_same_1_each_fullaug_sp3_s2","CFCDnew_swap_same_1_each_fullaug_sp3_s3","CFCDnew_swap_same_1_each_fullaug_sp4_s1","CFCDnew_swap_same_1_each_fullaug_sp4_s2","CFCDnew_swap_same_1_each_fullaug_sp4_s3","CFCDnew_swap_same_1_each_fullaug_sp5_s1","CFCDnew_swap_same_1_each_fullaug_sp5_s2","CFCDnew_swap_same_1_each_fullaug_sp5_s3"], "MLG 02 02": ["MLG_02_02_s1", "MLG_02_02_s2", "MLG_02_02_s3", "MLG_02_02_sp2_s1", "MLG_02_02_sp2_s2", "MLG_02_02_sp2_s3", "MLG_02_02_sp3_s1","MLG_02_02_sp3_s2","MLG_02_02_sp3_s3","MLG_02_02_sp4_s1","MLG_02_02_sp4_s2","MLG_02_02_sp4_s3","MLG_02_02_sp5_s1","MLG_02_02_sp5_s2","MLG_02_02_sp5_s3"], "MLG 05 05": ["MLG_05_05_s1", "MLG_05_05_s2", "MLG_05_05_s3", "MLG_05_05_sp2_s1", "MLG_05_05_sp2_s2", "MLG_05_05_sp2_s3", "MLG_05_05_sp3_s1","MLG_05_05_sp3_s2","MLG_05_05_sp3_s3","MLG_05_05_sp4_s1","MLG_05_05_sp4_s2","MLG_05_05_sp4_s3","MLG_05_05_sp5_s1","MLG_05_05_sp5_s2","MLG_05_05_sp5_s3"], "MLG 08 08": ["MLG_08_08_s1", "MLG_08_08_s2", "MLG_08_08_s3", "MLG_08_08_sp2_s1", "MLG_08_08_sp2_s2", "MLG_08_08_sp2_s3", "MLG_08_08_sp3_s1","MLG_08_08_sp3_s2","MLG_08_08_sp3_s3","MLG_08_08_sp4_s1","MLG_08_08_sp4_s2","MLG_08_08_sp4_s3","MLG_08_08_sp5_s1","MLG_08_08_sp5_s2","MLG_08_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name "MLG"



# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDIDnew_same_1_each_s1", "CFCDIDnew_same_1_each_s2", "CFCDIDnew_same_1_each_s3", "CFCDIDnew_same_1_each_sp2_s1", "CFCDIDnew_same_1_each_sp2_s2", "CFCDIDnew_same_1_each_sp2_s3", "CFCDIDnew_same_1_each_sp3_s1","CFCDIDnew_same_1_each_sp3_s2","CFCDIDnew_same_1_each_sp3_s3","CFCDIDnew_same_1_each_sp4_s1","CFCDIDnew_same_1_each_sp4_s2","CFCDIDnew_same_1_each_sp4_s3","CFCDIDnew_same_1_each_sp5_s1","CFCDIDnew_same_1_each_sp5_s2","CFCDIDnew_same_1_each_sp5_s3"],"CFCD": ["CFCDnew_swap_same_1_each_fullaug_s1", "CFCDnew_swap_same_1_each_fullaug_s2", "CFCDnew_swap_same_1_each_fullaug_s3", "CFCDnew_swap_same_1_each_fullaug_sp2_s1", "CFCDnew_swap_same_1_each_fullaug_sp2_s2", "CFCDnew_swap_same_1_each_fullaug_sp2_s3", "CFCDnew_swap_same_1_each_fullaug_sp3_s1","CFCDnew_swap_same_1_each_fullaug_sp3_s2","CFCDnew_swap_same_1_each_fullaug_sp3_s3","CFCDnew_swap_same_1_each_fullaug_sp4_s1","CFCDnew_swap_same_1_each_fullaug_sp4_s2","CFCDnew_swap_same_1_each_fullaug_sp4_s3","CFCDnew_swap_same_1_each_fullaug_sp5_s1","CFCDnew_swap_same_1_each_fullaug_sp5_s2","CFCDnew_swap_same_1_each_fullaug_sp5_s3"], "MSam": ["MSam_311_s1", "MSam_311_s2", "MSam_311_s3", "MSam_311_sp2_s1", "MSam_311_sp2_s2", "MSam_311_sp2_s3", "MSam_311_sp3_s1","MSam_311_sp3_s2","MSam_311_sp3_s3","MSam_311_sp4_s1","MSam_311_sp4_s2","MSam_311_sp4_s3","MSam_311_sp5_s1","MSam_311_sp5_s2","MSam_311_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MSam_311


# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"],"CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"], "MLG 08 08": ["MLG_08_08_s1", "MLG_08_08_s2", "MLG_08_08_s3", "MLG_08_08_sp2_s1", "MLG_08_08_sp2_s2", "MLG_08_08_sp2_s3", "MLG_08_08_sp3_s1","MLG_08_08_sp3_s2","MLG_08_08_sp3_s3","MLG_08_08_sp4_s1","MLG_08_08_sp4_s2","MLG_08_08_sp4_s3","MLG_08_08_sp5_s1","MLG_08_08_sp5_s2","MLG_08_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name delMLG


# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"],"CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"], "MSam 3, 1 temp, 1 var": ["MSam_311_s1", "MSam_311_s2", "MSam_311_s3", "MSam_311_sp2_s1", "MSam_311_sp2_s2", "MSam_311_sp2_s3", "MSam_311_sp3_s1","MSam_311_sp3_s2","MSam_311_sp3_s3","MSam_311_sp4_s1","MSam_311_sp4_s2","MSam_311_sp4_s3","MSam_311_sp5_s1","MSam_311_sp5_s2","MSam_311_sp5_s3"], "MSam 3, 0.3 temp, 1 var": ["MSam_3031_s1", "MSam_3031_s2", "MSam_3031_s3", "MSam_3031_sp2_s1", "MSam_3031_sp2_s2", "MSam_3031_sp2_s3", "MSam_3031_sp3_s1","MSam_3031_sp3_s2","MSam_3031_sp3_s3","MSam_3031_sp4_s1","MSam_3031_sp4_s2","MSam_3031_sp4_s3","MSam_3031_sp5_s1","MSam_3031_sp5_s2","MSam_3031_sp5_s3"], "MSam 3, 1 temp, 0.5 var": ["MSam_3105_s1", "MSam_3105_s2", "MSam_3105_s3", "MSam_3105_sp2_s1", "MSam_3105_sp2_s2", "MSam_3105_sp2_s3", "MSam_3105_sp3_s1","MSam_3105_sp3_s2","MSam_3105_sp3_s3","MSam_3105_sp4_s1","MSam_3105_sp4_s2","MSam_3105_sp4_s3","MSam_3105_sp5_s1","MSam_3105_sp5_s2","MSam_3105_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name M_Sampling


# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"],"CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"], "MG 08": ["MG_08_s1", "MG_08_s2", "MG_08_s3", "MG_08_sp2_s1", "MG_08_sp2_s2", "MG_08_sp2_s3", "MG_08_sp3_s1","MG_08_sp3_s2","MG_08_sp3_s3","MG_08_sp4_s1","MG_08_sp4_s2","MG_08_sp4_s3","MG_08_sp5_s1","MG_08_sp5_s2","MG_08_sp5_s3"], "MLG 08 08": ["MLG_08_08_s1", "MLG_08_08_s2", "MLG_08_08_s3", "MLG_08_08_sp2_s1", "MLG_08_08_sp2_s2", "MLG_08_08_sp2_s3", "MLG_08_08_sp3_s1","MLG_08_08_sp3_s2","MLG_08_08_sp3_s3","MLG_08_08_sp4_s1","MLG_08_08_sp4_s2","MLG_08_08_sp4_s3","MLG_08_08_sp5_s1","MLG_08_08_sp5_s2","MLG_08_08_sp5_s3"], "MG 08 01t": ["MG_01t_s1", "MG_01t_s2", "MG_01t_s3", "MG_01t_sp2_s1", "MG_01t_sp2_s2", "MG_01t_sp2_s3", "MG_01t_sp3_s1","MG_01t_sp3_s2","MG_01t_sp3_s3","MG_01t_sp4_s1","MG_01t_sp4_s2","MG_01t_sp4_s3","MG_01t_sp5_s1","MG_01t_sp5_s2","MG_01t_sp5_s3"], "MLG 08 08 01t": ["MLG_01t_s1", "MLG_01t_s2", "MLG_01t_s3", "MLG_01t_sp2_s1", "MLG_01t_sp2_s2", "MLG_01t_sp2_s3", "MLG_01t_sp3_s1","MLG_01t_sp3_s2","MLG_01t_sp3_s3","MLG_01t_sp4_s1","MLG_01t_sp4_s2","MLG_01t_sp4_s3","MLG_01t_sp5_s1","MLG_01t_sp5_s2","MLG_01t_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MLG_temps_redo

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"],"CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"], "MLG 08 08": ["MLG_08_08_s1", "MLG_08_08_s2", "MLG_08_08_s3", "MLG_08_08_sp2_s1", "MLG_08_08_sp2_s2", "MLG_08_08_sp2_s3", "MLG_08_08_sp3_s1","MLG_08_08_sp3_s2","MLG_08_08_sp3_s3","MLG_08_08_sp4_s1","MLG_08_08_sp4_s2","MLG_08_08_sp4_s3","MLG_08_08_sp5_s1","MLG_08_08_sp5_s2","MLG_08_08_sp5_s3"], "MLG 08 08 01t": ["MLG_01t_s1", "MLG_01t_s2", "MLG_01t_s3", "MLG_01t_sp2_s1", "MLG_01t_sp2_s2", "MLG_01t_sp2_s3", "MLG_01t_sp3_s1","MLG_01t_sp3_s2","MLG_01t_sp3_s3","MLG_01t_sp4_s1","MLG_01t_sp4_s2","MLG_01t_sp4_s3","MLG_01t_sp5_s1","MLG_01t_sp5_s2","MLG_01t_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MLG_temps


# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"],"CFCD": ["CFCD_s1", "CFCD_s2", "CFCD_s3", "CFCD_sp2_s1", "CFCD_sp2_s2", "CFCD_sp2_s3", "CFCD_sp3_s1","CFCD_sp3_s2","CFCD_sp3_s3","CFCD_sp4_s1","CFCD_sp4_s2","CFCD_sp4_s3","CFCD_sp5_s1","CFCD_sp5_s2","CFCD_sp5_s3"],"CFCD 0.2 Test": ["CFCD_0.2_s1", "CFCD_0.2_s2", "CFCD_0.2_s3", "CFCD_0.2_sp2_s1", "CFCD_0.2_sp2_s2", "CFCD_0.2_sp2_s3", "CFCD_0.2_sp3_s1","CFCD_0.2_sp3_s2","CFCD_0.2_sp3_s3","CFCD_0.2_sp4_s1","CFCD_0.2_sp4_s2","CFCD_0.2_sp4_s3","CFCD_0.2_sp5_s1","CFCD_0.2_sp5_s2","CFCD_0.2_sp5_s3"],"CFCDID 0.2 Test": ["CFCDID_0.2_s1", "CFCDID_0.2_s2", "CFCDID_0.2_s3", "CFCDID_0.2_sp2_s1", "CFCDID_0.2_sp2_s2", "CFCDID_0.2_sp2_s3", "CFCDID_0.2_sp3_s1","CFCDID_0.2_sp3_s2","CFCDID_0.2_sp3_s3","CFCDID_0.2_sp4_s1","CFCDID_0.2_sp4_s2","CFCDID_0.2_sp4_s3","CFCDID_0.2_sp5_s1","CFCDID_0.2_sp5_s2","CFCDID_0.2_sp5_s3"],"CFCD 0.2 7:3": ["CFCD_0.2_73_sp1_s1", "CFCD_0.2_73_sp1_s2", "CFCD_0.2_73_sp1_s3", "CFCD_0.2_73_sp2_s1", "CFCD_0.2_73_sp2_s2", "CFCD_0.2_73_sp2_s3", "CFCD_0.2_73_sp3_s1","CFCD_0.2_73_sp3_s2","CFCD_0.2_73_sp3_s3","CFCD_0.2_73_sp4_s1","CFCD_0.2_73_sp4_s2","CFCD_0.2_73_sp4_s3","CFCD_0.2_73_sp5_s1","CFCD_0.2_73_sp5_s2","CFCD_0.2_73_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDID0.273Test

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCD 0.2 7:3 sp1": ["CFCD_0.2_73_sp1_s1", "CFCD_0.2_73_sp1_s2", "CFCD_0.2_73_sp1_s3"], "CFCD 0.2 7:3 sp2": ["CFCD_0.2_73_sp2_s1", "CFCD_0.2_73_sp2_s2", "CFCD_0.2_73_sp2_s3"], "CFCD 0.2 7:3 sp3": ["CFCD_0.2_73_sp3_s1", "CFCD_0.2_73_sp3_s2", "CFCD_0.2_73_sp3_s3"], "CFCD 0.2 7:3 sp4": ["CFCD_0.2_73_sp4_s1", "CFCD_0.2_73_sp4_s2", "CFCD_0.2_73_sp4_s3"], "CFCD 0.2 7:3 sp5": ["CFCD_0.2_73_sp5_s1", "CFCD_0.2_73_sp5_s2", "CFCD_0.2_73_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDID0.273Splits

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID 0.2 7:3 sp1": ["CFCDID_0.2_sp1_s1", "CFCDID_0.2_sp1_s2", "CFCDID_0.2_sp1_s3"], "CFCDID 0.2 7:3 sp2": ["CFCDID_0.2_sp2_s1", "CFCDID_0.2_sp2_s2", "CFCDID_0.2_sp2_s3"], "CFCDID 0.2 7:3 sp3": ["CFCDID_0.2_sp3_s1", "CFCDID_0.2_sp3_s2", "CFCDID_0.2_sp3_s3"], "CFCDID 0.2 7:3 sp4": ["CFCDID_0.2_sp4_s1", "CFCDID_0.2_sp4_s2", "CFCDID_0.2_sp4_s3"], "CFCDID 0.2 7:3 sp5": ["CFCDID_0.2_sp5_s1", "CFCDID_0.2_sp5_s2", "CFCDID_0.2_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDID0.2Splits

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID 0.2 sp1": ["CFCDID_0.2_s1"], "CFCDID 0.2 sp1 reversed": ["CFCDID_0.2_reversed_sp1_s1"], "Redo": ["CFCDID_0.2_reversed_redo_sp1_s1"], "RedoRedo": ["CFCDID_0.2_reversed_redo2_sp1_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDIDReversed0.2SplitsRedo2fold1 --folds 1

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID 0.2 7:3 sp1": ["CFCDID_0.2_s1"], "CFCDID 0.2 7:3 sp2": ["CFCDID_0.2_sp2_s1"], "CFCDID 0.2 7:3 sp3": ["CFCDID_0.2_sp3_s1"], "CFCDID 0.2 7:3 sp4": ["CFCDID_0.2_sp4_s1"], "CFCDID 0.2 7:3 sp5": ["CFCDID_0.2_sp5_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDID0.2SplitsFolds --folds 1

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDIDOnly

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1"],"CFCDID Redo": ["CFCDID_redo_sp1_s1"], "CFCDID Redo 2": ["CFCDID_redo2_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name del

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDIDnew_same_1_each_s1"],"CFCD": ["CFCDnew_swap_same_1_each_fullaug_s1"], "CFCDID REDO": ["CFCDID_s1"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name del

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_s1", "CFCDID_s2", "CFCDID_s3", "CFCDID_sp2_s1", "CFCDID_sp2_s2", "CFCDID_sp2_s3", "CFCDID_sp3_s1","CFCDID_sp3_s2","CFCDID_sp3_s3","CFCDID_sp4_s1","CFCDID_sp4_s2","CFCDID_sp4_s3","CFCDID_sp5_s1","CFCDID_sp5_s2","CFCDID_sp5_s3"],"CFCDID": ["CFCDID_redo_sp1_s1", "CFCDID_redo_sp1_s2", "CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s1", "CFCDID_redo_sp2_s2", "CFCDID_redo_sp2_s3", "CFCDID_redo_sp3_s1","CFCDID_redo_sp3_s2","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s1","CFCDID_redo_sp4_s2","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s1","CFCDID_redo_sp5_s2","CFCDID_redo_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDIDNew

# Final Framework (is not same as old, but also good): python federated_train.py --gpus 0 --num_clients 3 --exp_code CFCDID_redo5 --no_verbose --split_dir chimera_3_5_2_same_1_each_balanced_0.5_0.5_nocd --num_rounds 40 --folds 5 --seed 1 --no_phases --max_epochs 3 --augmentations 0=features_1536_fixed 1=features_1536_fixed 2=features_1536_fixed --num_stages 2

# -> Now default -> python federated_train.py --gpus 0 --exp_code CFCDID_redo5 --no_verbose


# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"MLG": ["MLG_08_08_sp1_s3","MLG_08_08_sp2_s3","MLG_08_08_sp3_s3","MLG_08_08_sp4_s3","MLG_08_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MLG

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"ML": ["ML_08_sp1_s3","ML_08_sp2_s3","ML_08_sp3_s3","ML_08_sp4_s3","ML_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ML

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"MG": ["MG_08_sp1_s3","MG_08_sp2_s3","MG_08_sp3_s3","MG_08_sp4_s3","MG_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MG

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_redo_sp1_s1", "CFCDID_redo_sp1_s2", "CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s1", "CFCDID_redo_sp2_s2", "CFCDID_redo_sp2_s3", "CFCDID_redo_sp3_s1","CFCDID_redo_sp3_s2","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s1","CFCDID_redo_sp4_s2","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s1","CFCDID_redo_sp5_s2","CFCDID_redo_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDIDNew

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_redo_sp1_s1", "CFCDID_redo_sp1_s2", "CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s1", "CFCDID_redo_sp2_s2", "CFCDID_redo_sp2_s3", "CFCDID_redo_sp3_s1","CFCDID_redo_sp3_s2","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s1","CFCDID_redo_sp4_s2","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s1","CFCDID_redo_sp5_s2","CFCDID_redo_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CFCDIDCentralized

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3", "CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"], "ML 02 str10": ["ML_02_str10_sp1_s3", "ML_02_str10_sp2_s3","ML_02_str10_sp3_s3", "ML_02_str10_sp4_s3","ML_02_str10_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ML02str10

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"MG 02 t03": ["MG_02_t03_sp1_s3","MG_02_t03_sp2_s3","MG_02_t03_sp3_s3","MG_02_t03_sp4_s3","MG_02_t03_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MG02_t03

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"MG 02 t03": ["MG_02_t03_sp1_s3","MG_02_t03_sp2_s3","MG_02_t03_sp3_s3","MG_02_t03_sp4_s3","MG_02_t03_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MG02_t03

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3", "CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"], "ML 02 str10": ["ML_02_str10_sp1_s3", "ML_02_str10_sp2_s3","ML_02_str10_sp3_s3", "ML_02_str10_sp4_s3","ML_02_str10_sp5_s3"], "ML 08 str3": ["ML_08_str3_sp1_s3", "ML_08_str3_sp2_s3", "ML_08_str3_sp3_s3", "ML_08_str3_sp4_s3", "ML_08_str3_sp5_s3"], "ML 08": ["ML_08_sp1_s3","ML_08_sp2_s3","ML_08_sp3_s3","ML_08_sp4_s3","ML_08_sp5_s3"], "ML 01": ["ML_01_sp1_s3","ML_01_sp2_s3","ML_01_sp3_s3","ML_01_sp4_s3","ML_01_sp5_s3"], "ML_01_3str": ["ML_01_3str_sp1_s3","ML_01_3str_sp2_s3","ML_01_3str_sp3_s3","ML_01_3str_sp4_s3","ML_01_3str_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ML01-3str

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3", "CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"ML 08": ["ML_08_sp1_s3","ML_08_sp2_s3","ML_08_sp3_s3","ML_08_sp4_s3","ML_08_sp5_s3"], "ML 01": ["ML_01_sp1_s3","ML_01_sp2_s3","ML_01_sp3_s3","ML_01_sp4_s3","ML_01_sp5_s3"],"ML 00": ["ML_00_sp1_s3","ML_00_sp2_s3","ML_00_sp3_s3","ML_00_sp4_s3","ML_00_sp5_s3"] }' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ML00

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3", "CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"], "Centralized 1 stage": ["Centralized_1stage_sp1_s3", "Centralized_1stage_sp2_s3","Centralized_1stage_sp3_s3","Centralized_1stage_sp4_s3","Centralized_1stage_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name Centralized_1stage

# 

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"MG 08": ["MG_08_sp1_s3", "MG_08_sp2_s3", "MG_08_sp3_s3", "MG_08_sp4_s3","MG_08_sp5_s3"],"MG 00": ["MG_00_sp1_s3", "MG_00_sp2_s3", "MG_00_sp3_s3", "MG_00_sp4_s3","MG_00_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MG00

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID Seed 3": ["CFCDID_redo_sp1_s3", "CFCDID_redo_sp2_s3","CFCDID_redo_sp3_s3","CFCDID_redo_sp4_s3","CFCDID_redo_sp5_s3"],"MS 2 2 06": ["MS_2_2_06_sp1_s3", "MS_2_2_06_sp2_s3", "MS_2_2_06_sp3_s3", "MS_2_2_06_sp4_s3","MS_2_2_06_sp5_s3"],"MS 3 1 08": ["MS_3_1_08_sp1_s3", "MS_3_1_08_sp2_s3", "MS_3_1_08_sp3_s3", "MS_3_1_08_sp4_s3","MS_3_1_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name MS

# python plot/plot_crossfold_group_comparison.py --groups '{"Artificial CFCD ": ["art_CFCD_sp1_s3", "art_CFCD_sp2_s3","art_CFCD_sp3_s3","art_CFCD_sp4_s3","art_CFCD_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ArtCFCD

# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID ": ["art_CFCDID_sp1_s3", "art_CFCDID_sp2_s3","art_CFCDID_sp3_s3","art_CFCDID_sp4_s3","art_CFCDID_sp5_s3"], "Artificial CFCD ": ["art_CFCD_sp1_s3", "art_CFCD_sp2_s3","art_CFCD_sp3_s3","art_CFCD_sp4_s3","art_CFCD_sp5_s3"], "ML 08": ["ML_08_sp1_s3", "ML_08_sp2_s3","ML_08_sp3_s3","ML_08_sp4_s3","ML_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ArtCFCDID


# python plot/plot_crossfold_group_comparison.py --groups '{"CFCDID ": ["art_CFCDID_sp1_s3", "art_CFCDID_sp2_s3","art_CFCDID_sp3_s3","art_CFCDID_sp4_s3","art_CFCDID_sp5_s3"], "Artificial CFCD ": ["art_CFCD_sp1_s3", "art_CFCD_sp2_s3","art_CFCD_sp3_s3","art_CFCD_sp4_s3","art_CFCD_sp5_s3"], "CFCDID No Weighted Training": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "Artificial CFCD No Weighted Training": ["art_CFCD_no_weighted_training_sp1_s3", "art_CFCD_no_weighted_training_sp2_s3","art_CFCD_no_weighted_training_sp3_s3","art_CFCD_no_weighted_training_sp4_s3","art_CFCD_no_weighted_training_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ArtCFCDIDNoWeightedTraining

# python plot/plot_crossfold_group_comparison.py --groups '{"Artificial CFCD No Weighted Training": ["art_CFCD_no_weighted_training_sp1_s3", "art_CFCD_no_weighted_training_sp2_s3","art_CFCD_no_weighted_training_sp3_s3","art_CFCD_no_weighted_training_sp4_s3","art_CFCD_no_weighted_training_sp5_s3"], "CFCDID No Weighted Training": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "Artificial CFCD No Weighted Training": ["art_CFCD_no_weighted_training_sp1_s3", "art_CFCD_no_weighted_training_sp2_s3","art_CFCD_no_weighted_training_sp3_s3","art_CFCD_no_weighted_training_sp4_s3","art_CFCD_no_weighted_training_sp5_s3"], "70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "60 - 40": ["art_CFCD_no_weighted_training_6_4_sp1_s3", "art_CFCD_no_weighted_training_6_4_sp2_s3","art_CFCD_no_weighted_training_6_4_sp3_s3","art_CFCD_no_weighted_training_6_4_sp4_s3","art_CFCD_no_weighted_training_6_4_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name ArtCFCDIDNoWeightedTrainingNewSplitsAllFolds


# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "CD": ["art_CD_no_weighted_training_sp1_s3","art_CD_no_weighted_training_sp2_s3","art_CD_no_weighted_training_sp3_s3","art_CD_no_weighted_training_sp4_s3","art_CD_no_weighted_training_sp5_s3"], "CF 7-3": ["art_CF_no_weighted_training_7_3_sp1_s3","art_CF_no_weighted_training_7_3_sp2_s3","art_CF_no_weighted_training_7_3_sp3_s3","art_CF_no_weighted_training_7_3_sp4_s3","art_CF_no_weighted_training_7_3_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name CDCF_CD_CF

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "MG 02 02t": ["73MG_02_02t_sp1_s3","73MG_02_02t_sp2_s3","73MG_02_02t_sp3_s3","73MG_02_02t_sp4_s3","73MG_02_02t_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73MG02_02t

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "MS 2 2 08": ["73_MS_2_2_08_sp1_s3","73_MS_2_2_08_sp2_s3","73_MS_2_2_08_sp3_s3","73_MS_2_2_08_sp4_s3","73_MS_2_2_08_sp5_s3"], "ML 02 02str": ["73_ML_02_str2_sp1_s3","73_ML_02_str2_sp2_s3","73_ML_02_str2_sp3_s3","73_ML_02_str2_sp4_s3","73_ML_02_str2_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73MLMS

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Split 1": ["art_CFCD_no_weighted_training_7_3_sp1_s3"], "Split 2": ["art_CFCD_no_weighted_training_7_3_sp2_s3"], "Split 3": ["art_CFCD_no_weighted_training_7_3_sp3_s3"], "Split 4": ["art_CFCD_no_weighted_training_7_3_sp4_s3"], "Split 5": ["art_CFCD_no_weighted_training_7_3_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 70_30_Splits


# python federated_train.py --gpus 2 --num_clients 3 --exp_code 73_ML_02_str2 --no_verbose --split_dir chimera_3_5_2_0.2_0.7_0.3  --augmentations 0=features_1536_fixed 1=features_1536_fixed 2=Aug0_brightness_460 --debug --num_sampled 2 --temperature 0.5 --variance_scale 0.6

# python federated_train.py --gpus 3 --num_clients 3 --exp_code 73_MS2208_ML02_MG02 --no_verbose --split_dir chimera_3_5_2_0.2_0.7_0.3  --augmentations 0=features_1536_fixed 1=features_1536_fixed 2=Aug0_brightness_460 --debug  --num_sampled 2 --temperature 2 --variance_scale 0.8 --method_local --proto_adaptation_rate_client 0.2 --method_global --proto_adaptation_rate_server 0.2

# python federated_train.py --gpus 2 --num_clients 3 --exp_code 73_MS_2_05_06 --no_verbose --split_dir chimera_3_5_2_0.2_0.7_0.3  --augmentations 0=features_1536_fixed 1=features_1536_fixed 2=Aug0_brightness_460 --debug --num_sampled 2 --temperature 0.5 --variance_scale 0.6

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "MG 02 02t": ["73MG_02_02t_sp1_s3","73MG_02_02t_sp2_s3","73MG_02_02t_sp3_s3","73MG_02_02t_sp4_s3","73MG_02_02t_sp5_s3"], "ML 02 02str": ["73_ML_02_str2_sp1_s3","73_ML_02_str2_sp2_s3","73_ML_02_str2_sp3_s3","73_ML_02_str2_sp4_s3","73_ML_02_str2_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73MLMG


# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "MS 2 2 08": ["73_MS_2_2_08_sp1_s3","73_MS_2_2_08_sp2_s3","73_MS_2_2_08_sp3_s3","73_MS_2_2_08_sp4_s3","73_MS_2_2_08_sp5_s3"], "MSMLMG": ["73_MS2208_ML02_MG02_sp1_s3","73_MS2208_ML02_MG02_sp2_s3","73_MS2208_ML02_MG02_sp3_s3","73_MS2208_ML02_MG02_sp4_s3","73_MS2208_ML02_MG02_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73MS

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "ML 02 02str": ["73_ML_02_str2_sp1_s3","73_ML_02_str2_sp2_s3","73_ML_02_str2_sp3_s3","73_ML_02_str2_sp4_s3","73_ML_02_str2_sp5_s3"], "73_MG_02_005": ["73_MG_02_005_sp1_s3","73_MG_02_005_sp2_s3","73_MG_02_005_sp3_s3","73_MG_02_005_sp4_s3","73_MG_02_005_sp5_s3"],"73ML_02_4str": ["73ML_02_4str_sp1_s3","73ML_02_4str_sp2_s3","73ML_02_4str_sp3_s3","73ML_02_4str_sp4_s3","73ML_02_4str_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73NewMLMG

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 08": ["73_MS_2_2_08_sp1_s3","73_MS_2_2_08_sp2_s3","73_MS_2_2_08_sp3_s3","73_MS_2_2_08_sp4_s3","73_MS_2_2_08_sp5_s3"], "73 MS 3 3 08": ["73_MS3308_sp1_s3","73_MS3308_sp2_s3","73_MS3308_sp3_s3","73_MS3308_sp4_s3","73_MS3308_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73NewMS

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 08": ["73_MS_2_2_08_sp1_s3","73_MS_2_2_08_sp2_s3","73_MS_2_2_08_sp3_s3","73_MS_2_2_08_sp4_s3","73_MS_2_2_08_sp5_s3"], "73 MS 1 1 08": ["73_MS_1_1_08_sp1_s3","73_MS_1_1_08_sp2_s3","73_MS_1_1_08_sp3_s3","73_MS_1_1_08_sp4_s3","73_MS_1_1_08_sp5_s3"],"73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"],"73 MS 2 0.5 08": ["73_MS_2_0.5_08_sp1_s3","73_MS_2_0.5_08_sp2_s3","73_MS_2_0.5_08_sp3_s3","73_MS_2_0.5_08_sp4_s3","73_MS_2_0.5_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73NewMSs

# python plot/plot_crossfold_group_comparison.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 08": ["73_MS_2_2_08_sp1_s3","73_MS_2_2_08_sp2_s3","73_MS_2_2_08_sp3_s3","73_MS_2_2_08_sp4_s3","73_MS_2_2_08_sp5_s3"], "73 MS 1 1 08": ["73_MS_1_1_08_sp1_s3","73_MS_1_1_08_sp2_s3","73_MS_1_1_08_sp3_s3","73_MS_1_1_08_sp4_s3","73_MS_1_1_08_sp5_s3"],"73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"],"73 MS 2 0.5 08": ["73_MS_2_0.5_08_sp1_s3","73_MS_2_0.5_08_sp2_s3","73_MS_2_0.5_08_sp3_s3","73_MS_2_0.5_08_sp4_s3","73_MS_2_0.5_08_sp5_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name 73NewMSs

# 73_ML_05_str3
# 73_ML_01_str5
# 73_ML_03_str05
# python plot/plot_only_test.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73_ML_05_str3": ["73_ML_05_str3_sp1_s1","73_ML_05_str3_sp2_s1", "73_ML_05_str3_sp3_s1", "73_ML_05_str3_sp4_s1", "73_ML_05_str3_sp5_s1", "73_ML_05_str3_sp1_s2","73_ML_05_str3_sp2_s2", "73_ML_05_str3_sp3_s2", "73_ML_05_str3_sp4_s2", "73_ML_05_str3_sp5_s2"], "73_ML_01_str5": ["73_ML_01_str5_sp1_s1","73_ML_01_str5_sp2_s1", "73_ML_01_str5_sp3_s1", "73_ML_01_str5_sp4_s1", "73_ML_01_str5_sp5_s1", "73_ML_01_str5_sp1_s2","73_ML_01_str5_sp2_s2", "73_ML_01_str5_sp3_s2", "73_ML_01_str5_sp4_s2", "73_ML_01_str5_sp5_s2"], "73_ML_03_str05": ["73_ML_03_str05_sp1_s1","73_ML_03_str05_sp2_s1", "73_ML_03_str05_sp3_s1", "73_ML_03_str05_sp4_s1", "73_ML_03_str05_sp5_s1", "73_ML_03_str05_sp1_s2","73_ML_03_str05_sp2_s2", "73_ML_03_str05_sp3_s2", "73_ML_03_str05_sp4_s2", "73_ML_03_str05_sp5_s2"]}' --name ex1


# 73_MG_05_01t
# 73_MG_03_05t
# 73_MG_01_002t
# python plot/plot_only_test.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73_MG_05_01t": ["73_MG_05_01t_sp1_s1","73_MG_05_01t_sp2_s1", "73_MG_05_01t_sp3_s1", "73_MG_05_01t_sp4_s1", "73_MG_05_01t_sp5_s1", "73_MG_05_01t_sp1_s2","73_MG_05_01t_sp2_s2", "73_MG_05_01t_sp3_s2", "73_MG_05_01t_sp4_s2", "73_MG_05_01t_sp5_s2"], "73_MG_03_05t": ["73_MG_03_05t_sp1_s1","73_MG_03_05t_sp2_s1", "73_MG_03_05t_sp3_s1", "73_MG_03_05t_sp4_s1", "73_MG_03_05t_sp5_s1", "73_MG_03_05t_sp1_s2","73_MG_03_05t_sp2_s2", "73_MG_03_05t_sp3_s2", "73_MG_03_05t_sp4_s2", "73_MG_03_05t_sp5_s2"]}' --name ex2


# 73_MS_1_05_08
# 73_MS_4_05_08
# 73_MS_2_3_06
# python plot/plot_only_test.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73_MS_1_05_08": ["73_MS_1_05_08_sp1_s1","73_MS_1_05_08_sp2_s1", "73_MS_1_05_08_sp3_s1", "73_MS_1_05_08_sp4_s1", "73_MS_1_05_08_sp5_s1", "73_MS_1_05_08_sp1_s2","73_MS_1_05_08_sp2_s2", "73_MS_1_05_08_sp3_s2", "73_MS_1_05_08_sp4_s2", "73_MS_1_05_08_sp5_s2"], "73_MS_4_05_08": ["73_MS_4_05_08_sp1_s1","73_MS_4_05_08_sp2_s1", "73_MS_4_05_08_sp3_s1", "73_MS_4_05_08_sp4_s1", "73_MS_4_05_08_sp5_s1", "73_MS_4_05_08_sp1_s2","73_MS_4_05_08_sp2_s2", "73_MS_4_05_08_sp3_s2", "73_MS_4_05_08_sp4_s2", "73_MS_4_05_08_sp5_s2"], "73_MS_2_3_06": ["73_MS_2_3_06_sp1_s1","73_MS_2_3_06_sp2_s1", "73_MS_2_3_06_sp3_s1", "73_MS_2_3_06_sp4_s1", "73_MS_2_3_06_sp5_s1", "73_MS_2_3_06_sp1_s2","73_MS_2_3_06_sp2_s2", "73_MS_2_3_06_sp3_s2", "73_MS_2_3_06_sp4_s2", "73_MS_2_3_06_sp5_s2"]}' --name ex3

# repr_ML_05_str1
# repr_ML_03_str4
# repr_ML_01_str2
# repr_ML_08_str05
# python plot/plot_only_test.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s1", "art_CFCD_no_weighted_training_7_3_sp2_s1","art_CFCD_no_weighted_training_7_3_sp3_s1","art_CFCD_no_weighted_training_7_3_sp4_s1","art_CFCD_no_weighted_training_7_3_sp5_s1","art_CFCD_no_weighted_training_7_3_sp1_s2", "art_CFCD_no_weighted_training_7_3_sp2_s2","art_CFCD_no_weighted_training_7_3_sp3_s2","art_CFCD_no_weighted_training_7_3_sp4_s2","art_CFCD_no_weighted_training_7_3_sp5_s2"], "repr_ML_05_str1": ["repr_ML_05_str1_sp1_s1","repr_ML_05_str1_sp2_s1", "repr_ML_05_str1_sp3_s1", "repr_ML_05_str1_sp4_s1", "repr_ML_05_str1_sp5_s1", "repr_ML_05_str1_sp1_s2","repr_ML_05_str1_sp2_s2", "repr_ML_05_str1_sp3_s2", "repr_ML_05_str1_sp4_s2", "repr_ML_05_str1_sp5_s2"], "repr_ML_03_str4": ["repr_ML_03_str4_sp1_s1","repr_ML_03_str4_sp2_s1", "repr_ML_03_str4_sp3_s1", "repr_ML_03_str4_sp4_s1", "repr_ML_03_str4_sp5_s1", "repr_ML_03_str4_sp1_s2","repr_ML_03_str4_sp2_s2", "repr_ML_03_str4_sp3_s2", "repr_ML_03_str4_sp4_s2", "repr_ML_03_str4_sp5_s2"], "repr_ML_01_str2": ["repr_ML_01_str2_sp1_s1","repr_ML_01_str2_sp2_s1", "repr_ML_01_str2_sp3_s1", "repr_ML_01_str2_sp4_s1", "repr_ML_01_str2_sp5_s1", "repr_ML_01_str2_sp1_s2","repr_ML_01_str2_sp2_s2", "repr_ML_01_str2_sp3_s2", "repr_ML_01_str2_sp4_s2", "repr_ML_01_str2_sp5_s2"], "repr_ML_08_str05": ["repr_ML_08_str05_sp1_s1","repr_ML_08_str05_sp2_s1", "repr_ML_08_str05_sp3_s1", "repr_ML_08_str05_sp4_s1", "repr_ML_08_str05_sp5_s1", "repr_ML_08_str05_sp1_s2","repr_ML_08_str05_sp2_s2", "repr_ML_08_str05_sp3_s2", "repr_ML_08_str05_sp4_s2", "repr_ML_08_str05_sp5_s2"]}' --name ex4




# repr_MG_05_02t
# repr_MG_01_005t
# repr_MG_03_01t
# repr_MS_1_05_02
# python plot/plot_only_test.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "repr_MG_05_02t": ["repr_MG_05_02t_sp1_s1","repr_MG_05_02t_sp2_s1", "repr_MG_05_02t_sp3_s1", "repr_MG_05_02t_sp4_s1", "repr_MG_05_02t_sp5_s1", "repr_MG_05_02t_sp1_s2","repr_MG_05_02t_sp2_s2", "repr_MG_05_02t_sp3_s2", "repr_MG_05_02t_sp4_s2", "repr_MG_05_02t_sp5_s2"], "repr_MG_01_005t": ["repr_MG_01_005t_sp1_s1","repr_MG_01_005t_sp2_s1", "repr_MG_01_005t_sp3_s1", "repr_MG_01_005t_sp4_s1", "repr_MG_01_005t_sp5_s1", "repr_MG_01_005t_sp1_s2","repr_MG_01_005t_sp2_s2", "repr_MG_01_005t_sp3_s2", "repr_MG_01_005t_sp4_s2", "repr_MG_01_005t_sp5_s2"]}' --name ex5


# repr_MLMG_05_2str_01_01t
# repr_MLMG_03_4str_03_02t
# repr_MLMG_01_1str_05_005t
# repr_MLMS_03_2str_2s_1_02
# python plot/plot_only_test.py --groups '{"70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "repr_MLMG_05_2str_01_01t": ["repr_MLMG_05_2str_01_01t_sp1_s1","repr_MLMG_05_2str_01_01t_sp2_s1", "repr_MLMG_05_2str_01_01t_sp3_s1", "repr_MLMG_05_2str_01_01t_sp4_s1", "repr_MLMG_05_2str_01_01t_sp5_s1", "repr_MLMG_05_2str_01_01t_sp1_s2","repr_MLMG_05_2str_01_01t_sp2_s2", "repr_MLMG_05_2str_01_01t_sp3_s2", "repr_MLMG_05_2str_01_01t_sp4_s2", "repr_MLMG_05_2str_01_01t_sp5_s2"], "repr_MLMG_03_4str_03_02t": ["repr_MLMG_03_4str_03_02t_sp1_s1","repr_MLMG_03_4str_03_02t_sp2_s1", "repr_MLMG_03_4str_03_02t_sp3_s1", "repr_MLMG_03_4str_03_02t_sp4_s1", "repr_MLMG_03_4str_03_02t_sp5_s1", "repr_MLMG_03_4str_03_02t_sp1_s2","repr_MLMG_03_4str_03_02t_sp2_s2", "repr_MLMG_03_4str_03_02t_sp3_s2", "repr_MLMG_03_4str_03_02t_sp4_s2", "repr_MLMG_03_4str_03_02t_sp5_s2"], "repr_MLMG_01_1str_05_005t": ["repr_MLMG_01_1str_05_005t_sp1_s1","repr_MLMG_01_1str_05_005t_sp2_s1", "repr_MLMG_01_1str_05_005t_sp3_s1", "repr_MLMG_01_1str_05_005t_sp4_s1", "repr_MLMG_01_1str_05_005t_sp5_s1", "repr_MLMG_01_1str_05_005t_sp1_s2","repr_MLMG_01_1str_05_005t_sp2_s2", "repr_MLMG_01_1str_05_005t_sp3_s2", "repr_MLMG_01_1str_05_005t_sp4_s2", "repr_MLMG_01_1str_05_005t_sp5_s2"], "repr_MLMS_03_2str_2s_1_02": ["repr_MLMS_03_2str_2s_1_02_sp1_s1","repr_MLMS_03_2str_2s_1_02_sp2_s1", "repr_MLMS_03_2str_2s_1_02_sp3_s1", "repr_MLMS_03_2str_2s_1_02_sp4_s1", "repr_MLMS_03_2str_2s_1_02_sp5_s1", "repr_MLMS_03_2str_2s_1_02_sp1_s2","repr_MLMS_03_2str_2s_1_02_sp2_s2", "repr_MLMS_03_2str_2s_1_02_sp3_s2", "repr_MLMS_03_2str_2s_1_02_sp4_s2"]}' --name ex6

# python federated_train.py --gpus 3 --num_clients 3 --exp_code 73_MS204_ML02s2_MG02t05 --no_verbose --split_dir chimera_3_5_2_0.2_0.7_0.3  --augmentations 0=features_1536_fixed 1=features_1536_fixed 2=Aug0_brightness_460 --debug --num_sampled 2 --variance_scale 0.4 --method_local --strictness 2 --proto_adaptation_rate_client 0.2 --method_global --temperature 0.5 --proto_adaptation_rate_server 0.2





# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"]}' --name MSMethod

# python plot/plot_only_test.py --groups '{"CFCDID": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "80 - 20": ["art_CFCD_no_weighted_training_sp1_s3", "art_CFCD_no_weighted_training_sp2_s3","art_CFCD_no_weighted_training_sp3_s3","art_CFCD_no_weighted_training_sp4_s3","art_CFCD_no_weighted_training_sp5_s3"],  "70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "60 - 40": ["art_CFCD_no_weighted_training_6_4_sp1_s3", "art_CFCD_no_weighted_training_6_4_sp2_s3","art_CFCD_no_weighted_training_6_4_sp3_s3","art_CFCD_no_weighted_training_6_4_sp4_s3","art_CFCD_no_weighted_training_6_4_sp5_s3"]}' --name del


# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s1", "art_CFCDID_no_weighted_training_sp2_s1","art_CFCDID_no_weighted_training_sp3_s1","art_CFCDID_no_weighted_training_sp4_s1","art_CFCDID_no_weighted_training_sp5_s1","art_CFCDID_no_weighted_training_sp1_s2", "art_CFCDID_no_weighted_training_sp2_s2","art_CFCDID_no_weighted_training_sp3_s2","art_CFCDID_no_weighted_training_sp4_s2","art_CFCDID_no_weighted_training_sp5_s2","art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s1", "art_CFCD_no_weighted_training_7_3_sp2_s1","art_CFCD_no_weighted_training_7_3_sp3_s1","art_CFCD_no_weighted_training_7_3_sp4_s1","art_CFCD_no_weighted_training_7_3_sp5_s1", "art_CFCD_no_weighted_training_7_3_sp1_s2", "art_CFCD_no_weighted_training_7_3_sp2_s2","art_CFCD_no_weighted_training_7_3_sp3_s2","art_CFCD_no_weighted_training_7_3_sp4_s2","art_CFCD_no_weighted_training_7_3_sp5_s2", "art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s1","73_MS_2_2_04_sp2_s1","73_MS_2_2_04_sp3_s1","73_MS_2_2_04_sp4_s1","73_MS_2_2_04_sp5_s1", "73_MS_2_2_04_sp1_s2","73_MS_2_2_04_sp2_s2","73_MS_2_2_04_sp3_s2","73_MS_2_2_04_sp4_s2","73_MS_2_2_04_sp5_s2", "73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "73 MS 2 04 01 01 Adaption": ["73_MS_2_04_0101_sp1_s1","73_MS_2_04_0101_sp2_s1","73_MS_2_04_0101_sp3_s1","73_MS_2_04_0101_sp4_s1","73_MS_2_04_0101_sp5_s1", "73_MS_2_04_0101_sp1_s2","73_MS_2_04_0101_sp2_s2","73_MS_2_04_0101_sp3_s2","73_MS_2_04_0101_sp4_s2","73_MS_2_04_0101_sp5_s2", "73_MS_2_04_0101_sp1_s3","73_MS_2_04_0101_sp2_s3","73_MS_2_04_0101_sp3_s3","73_MS_2_04_0101_sp4_s3","73_MS_2_04_0101_sp5_s3"]}' --name MSMethodFullSeeds01Adaption

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s1", "art_CFCDID_no_weighted_training_sp2_s1","art_CFCDID_no_weighted_training_sp3_s1","art_CFCDID_no_weighted_training_sp4_s1","art_CFCDID_no_weighted_training_sp5_s1","art_CFCDID_no_weighted_training_sp1_s2", "art_CFCDID_no_weighted_training_sp2_s2","art_CFCDID_no_weighted_training_sp3_s2","art_CFCDID_no_weighted_training_sp4_s2","art_CFCDID_no_weighted_training_sp5_s2","art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "60 - 40": ["art_CFCD_no_weighted_training_6_4_sp1_s1", "art_CFCD_no_weighted_training_6_4_sp2_s1","art_CFCD_no_weighted_training_6_4_sp3_s1","art_CFCD_no_weighted_training_6_4_sp4_s1","art_CFCD_no_weighted_training_6_4_sp5_s1", "art_CFCD_no_weighted_training_6_4_sp1_s2", "art_CFCD_no_weighted_training_6_4_sp2_s2","art_CFCD_no_weighted_training_6_4_sp3_s2","art_CFCD_no_weighted_training_6_4_sp4_s2","art_CFCD_no_weighted_training_6_4_sp5_s2", "art_CFCD_no_weighted_training_6_4_sp1_s3", "art_CFCD_no_weighted_training_6_4_sp2_s3","art_CFCD_no_weighted_training_6_4_sp3_s3","art_CFCD_no_weighted_training_6_4_sp4_s3","art_CFCD_no_weighted_training_6_4_sp5_s3"], "70 - 30": ["art_CFCD_no_weighted_training_7_3_sp1_s1", "art_CFCD_no_weighted_training_7_3_sp2_s1","art_CFCD_no_weighted_training_7_3_sp3_s1","art_CFCD_no_weighted_training_7_3_sp4_s1","art_CFCD_no_weighted_training_7_3_sp5_s1", "art_CFCD_no_weighted_training_7_3_sp1_s2", "art_CFCD_no_weighted_training_7_3_sp2_s2","art_CFCD_no_weighted_training_7_3_sp3_s2","art_CFCD_no_weighted_training_7_3_sp4_s2","art_CFCD_no_weighted_training_7_3_sp5_s2", "art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "75 - 25": ["art_CFCD_no_weighted_training_75_26_sp1_s1", "art_CFCD_no_weighted_training_75_26_sp2_s1","art_CFCD_no_weighted_training_75_26_sp3_s1","art_CFCD_no_weighted_training_75_26_sp4_s1","art_CFCD_no_weighted_training_75_26_sp5_s1", "art_CFCD_no_weighted_training_75_26_sp1_s2", "art_CFCD_no_weighted_training_75_26_sp2_s2","art_CFCD_no_weighted_training_75_26_sp3_s2","art_CFCD_no_weighted_training_75_26_sp4_s2","art_CFCD_no_weighted_training_75_26_sp5_s2", "art_CFCD_no_weighted_training_75_26_sp1_s3", "art_CFCD_no_weighted_training_75_26_sp2_s3","art_CFCD_no_weighted_training_75_26_sp3_s3","art_CFCD_no_weighted_training_75_26_sp4_s3","art_CFCD_no_weighted_training_75_26_sp5_s3"], "80 - 20": ["art_CFCD_no_weighted_training_sp1_s1", "art_CFCD_no_weighted_training_sp2_s1","art_CFCD_no_weighted_training_sp3_s1","art_CFCD_no_weighted_training_sp4_s1","art_CFCD_no_weighted_training_sp5_s1", "art_CFCD_no_weighted_training_sp1_s2", "art_CFCD_no_weighted_training_sp2_s2","art_CFCD_no_weighted_training_sp3_s2","art_CFCD_no_weighted_training_sp4_s2","art_CFCD_no_weighted_training_sp5_s2", "art_CFCD_no_weighted_training_sp1_s3", "art_CFCD_no_weighted_training_sp2_s3","art_CFCD_no_weighted_training_sp3_s3","art_CFCD_no_weighted_training_sp4_s3","art_CFCD_no_weighted_training_sp5_s3"]}' --name 64-73-82+

# python plot/plot_only_test.py --groups '{"1 Stage FL": ["1_stage_FL_sp1_s1", "1_stage_FL_sp2_s1","1_stage_FL_sp3_s1","1_stage_FL_sp4_s1","1_stage_FL_sp5_s1","1_stage_FL_sp1_s2", "1_stage_FL_sp2_s2","1_stage_FL_sp3_s2","1_stage_FL_sp4_s2","1_stage_FL_sp5_s2","1_stage_FL_sp1_s3", "1_stage_FL_sp2_s3","1_stage_FL_sp3_s3","1_stage_FL_sp4_s3","1_stage_FL_sp5_s3"]}' --name FLvsCentralized

# python plot/plot_clean_comparison.py --groups '{"1 Stage FL": ["1_stage_FL_sp1_s3"], "Centralized": ["1_stage_Centralized_sp1_s3"], "FL No Tuning": ["FL_baseline_no_paramtuning_sp1_s3"]}' --name FLvsCentralized

# python plot/plot_crossfold_group_comparison.py --groups '{"1 Stage FL": ["1_stage_FL_sp1_s3"], "Centralized": ["1_stage_Centralized_sp1_s3"]}' --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std --name FLvsCentralized


# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "73 MS 2 04 01 01 Adaption": ["73_MS_2_04_0101_sp1_s3","73_MS_2_04_0101_sp2_s3","73_MS_2_04_0101_sp3_s3","73_MS_2_04_0101_sp4_s3","73_MS_2_04_0101_sp5_s3"], "73 MS 2 04 01 01 With MG/ML activated": ["73_MS_2_04_redo_sp1_s3","73_MS_2_04_redo_sp2_s3","73_MS_2_04_redo_sp3_s3","73_MS_2_04_redo_sp4_s3","73_MS_2_04_redo_sp5_s3"]}' --name MSMethodFullSeeds01Adaption2

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "FedProx": ["fedprox_sp1_s3","fedprox_sp2_s3","fedprox_sp3_s3","fedprox_sp4_s3","fedprox_sp5_s3"]}' --name FedProx

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "FedAvgM": ["FedAvgM00105_sp1_s3","FedAvgM00105_sp2_s3","FedAvgM00105_sp3_s3","FedAvgM00105_sp4_s3","FedAvgM00105_sp5_s3"]}' --name FedAvgM2

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "FedAvgM": ["FedAvgM0105_sp1_s3","FedAvgM0105_sp2_s3","FedAvgM0105_sp3_s3","FedAvgM0105_sp4_s3","FedAvgM0105_sp5_s3"]}' --name FedAvgM3

# python plot/plot_only_test.py --groups '{"CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "FedAvgM 001 05": ["FedAvgM00105_sp1_s3","FedAvgM00105_sp2_s3","FedAvgM00105_sp3_s3","FedAvgM00105_sp4_s3","FedAvgM00105_sp5_s3"], "FedAvgM 01 05": ["FedAvgM0105_sp1_s3","FedAvgM0105_sp2_s3","FedAvgM0105_sp3_s3","FedAvgM0105_sp4_s3","FedAvgM0105_sp5_s3"], "FedAvgM 05 09": ["FedAvgM_sp1_s3","FedAvgM_sp2_s3","FedAvgM_sp3_s3","FedAvgM_sp4_s3","FedAvgM_sp5_s3"], "FedAvgM 005 06": ["FedAvgM00506_sp1_s3","FedAvgM00506_sp2_s3","FedAvgM00506_sp3_s3","FedAvgM00506_sp4_s3","FedAvgM00506_sp5_s3"]}' --name FedAvgMAll

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"],  "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "FL No Param Tuning": ["FL_baseline_no_paramtuning_sp1_s3", "FL_baseline_no_paramtuning_sp2_s3", "FL_baseline_no_paramtuning_sp3_s3", "FL_baseline_no_paramtuning_sp4_s3", "FL_baseline_no_paramtuning_sp5_s3"]}' --name FLNoParamTuning

# python plot/plot_crossfold_group_comparison.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3"], "FL No Param Tuning": ["FL_baseline_no_paramtuning_sp1_s3"]}' --name FL No Param Tuning 0 --folds 0 --submodel MM --metric_filter all --show_full_training --smooth_window 5 --show_std


# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Global T=0.5 Adapt=0.3": ["73_MG_03_05t_sp1_s3","73_MG_03_05t_sp2_s3","73_MG_03_05t_sp3_s3","73_MG_03_05t_sp4_s3","73_MG_03_05t_sp5_s3"], "Global T=0.05 Adapt=0.2": ["73_MG_02_005_sp1_s3","73_MG_02_005_sp2_s3","73_MG_02_005_sp3_s3","73_MG_02_005_sp4_s3","73_MG_02_005_sp5_s3"]}' --name MG

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Global T=0.5 Adapt=0.3": ["73_MG_03_05t_sp1_s3","73_MG_03_05t_sp2_s3","73_MG_03_05t_sp3_s3","73_MG_03_05t_sp4_s3","73_MG_03_05t_sp5_s3"], "Global T=0.05 Adapt=0.2": ["73_MG_02_005_sp1_s3","73_MG_02_005_sp2_s3","73_MG_02_005_sp3_s3","73_MG_02_005_sp4_s3","73_MG_02_005_sp5_s3"], "Global T=0.2 Adapat=0.2": ["73MG_02_02t_sp1_s3","73MG_02_02t_sp2_s3","73MG_02_02t_sp3_s3","73MG_02_02t_sp4_s3", "73MG_02_02t_sp5_s3"]}' --name MG2

#73_ML_02_2str_sp1_s3, 73_ML_02_str1_sp1_s3, 73_ML_02_str2_sp1_s3

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Local s=2 Adapt=0.2": ["73_ML_02_2str_sp1_s3","73_ML_02_2str_sp2_s3","73_ML_02_2str_sp3_s3","73_ML_02_2str_sp4_s3","73_ML_02_2str_sp5_s3"], "Local s=1 Adapt=0.2": ["73_ML_02_str1_sp1_s3","73_ML_02_str1_sp2_s3","73_ML_02_str1_sp3_s3","73_ML_02_str1_sp4_s3","73_ML_02_str1_sp5_s3"]}' --name ML


# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Generative Replay 2 Samples, γ=0.4": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "Generative Replay 4 Samples, γ=0.8": ["73_MS_4_05_08_sp1_s2","73_MS_4_05_08_sp2_s2","73_MS_4_05_08_sp3_s2","73_MS_4_05_08_sp4_s2","73_MS_4_05_08_sp5_s2"]}' --name MS

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Combined Method": ["73_MS204_ML02s2_MG02t05_sp1_s3","73_MS204_ML02s2_MG02t05_sp2_s3","73_MS204_ML02s2_MG02t05_sp3_s3","73_MS204_ML02s2_MG02t05_sp4_s3","73_MS204_ML02s2_MG02t05_sp5_s3"]}' --name MSMLMG

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Combined Method": ["73_MS204_ML02s2_MG02t05_sp1_s3","73_MS204_ML02s2_MG02t05_sp2_s3","73_MS204_ML02s2_MG02t05_sp3_s3","73_MS204_ML02s2_MG02t05_sp4_s3","73_MS204_ML02s2_MG02t05_sp5_s3"]}' --name MSMLMG


# python plot/plot_clean_comparison.py --groups '{"Federated Learning": ["1_stage_FL_sp1_s1", "1_stage_FL_sp2_s1","1_stage_FL_sp3_s1","1_stage_FL_sp4_s1","1_stage_FL_sp5_s1","1_stage_FL_sp1_s2", "1_stage_FL_sp2_s2","1_stage_FL_sp3_s2","1_stage_FL_sp4_s2","1_stage_FL_sp5_s2","1_stage_FL_sp1_s3", "1_stage_FL_sp2_s3","1_stage_FL_sp3_s3","1_stage_FL_sp4_s3","1_stage_FL_sp5_s3"], "Centralized": ["1_stage_Centralized_sp1_s1", "1_stage_Centralized_sp2_s1","1_stage_Centralized_sp3_s1","1_stage_Centralized_sp4_s1","1_stage_Centralized_sp5_s1","1_stage_Centralized_sp1_s2", "1_stage_Centralized_sp2_s2","1_stage_Centralized_sp3_s2","1_stage_Centralized_sp4_s2","1_stage_Centralized_sp5_s2","1_stage_Centralized_sp1_s3", "1_stage_Centralized_sp2_s3","1_stage_Centralized_sp3_s3","1_stage_Centralized_sp4_s3","1_stage_Centralized_sp5_s3"]}' --name FLvsCentralized_clean_2


# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Feat-based Gen Sampling": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "Repr-based Gen Sampling": ["repr_MS_2_2_04_sp1_s3","repr_MS_2_2_04_sp2_s3","repr_MS_2_2_04_sp3_s3","repr_MS_2_2_04_sp4_s3","repr_MS_2_2_04_sp5_s3"]}' --name ReprvsFeat

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "73 MS 2 2 04": ["73_MS_2_2_04_sp1_s3","73_MS_2_2_04_sp2_s3","73_MS_2_2_04_sp3_s3","73_MS_2_2_04_sp4_s3","73_MS_2_2_04_sp5_s3"], "CF Solo": ["73_CF_Solo_sp1_s3","73_CF_Solo_sp2_s3","73_CF_Solo_sp3_s3","73_CF_Solo_sp4_s3","73_CF_Solo_sp5_s3"]}' --name CFSolo

# python plot/plot_with_drift.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s1", "art_CFCDID_no_weighted_training_sp2_s1","art_CFCDID_no_weighted_training_sp3_s1","art_CFCDID_no_weighted_training_sp4_s1","art_CFCDID_no_weighted_training_sp5_s1","art_CFCDID_no_weighted_training_sp1_s2", "art_CFCDID_no_weighted_training_sp2_s2","art_CFCDID_no_weighted_training_sp3_s2","art_CFCDID_no_weighted_training_sp4_s2","art_CFCDID_no_weighted_training_sp5_s2","art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s1", "art_CFCD_no_weighted_training_7_3_sp2_s1","art_CFCD_no_weighted_training_7_3_sp3_s1","art_CFCD_no_weighted_training_7_3_sp4_s1","art_CFCD_no_weighted_training_7_3_sp5_s1", "art_CFCD_no_weighted_training_7_3_sp1_s2", "art_CFCD_no_weighted_training_7_3_sp2_s2","art_CFCD_no_weighted_training_7_3_sp3_s2","art_CFCD_no_weighted_training_7_3_sp4_s2","art_CFCD_no_weighted_training_7_3_sp5_s2", "art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "CF Solo": ["73_CF_Solo_sp1_s1","73_CF_Solo_sp2_s1","73_CF_Solo_sp3_s1","73_CF_Solo_sp4_s1","73_CF_Solo_sp5_s1","73_CF_Solo_sp1_s2","73_CF_Solo_sp2_s2","73_CF_Solo_sp3_s2","73_CF_Solo_sp4_s2","73_CF_Solo_sp5_s2","73_CF_Solo_sp1_s3","73_CF_Solo_sp2_s3","73_CF_Solo_sp3_s3","73_CF_Solo_sp4_s3","73_CF_Solo_sp5_s3"]}' --name CFSolo3Seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "Local s=2 Adapt=0.2": ["73_ML_02_2str_sp1_s3","73_ML_02_2str_sp2_s3","73_ML_02_2str_sp3_s3","73_ML_02_2str_sp4_s3","73_ML_02_2str_sp5_s3"], "Local s=1 Adapt=0.2": ["73_ML_02_str1_sp1_s3","73_ML_02_str1_sp2_s3","73_ML_02_str1_sp3_s3","73_ML_02_str1_sp4_s3","73_ML_02_str1_sp5_s3"]}' --name ML

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Local s=2 Adapt=0.2": ["73_ML_02_2str"], "Local s=1 Adapt=0.2": ["73_ML_02_str1"]}' --name ML --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Global T=0.5 Adapt=0.3": ["73_MG_03_05t"], "Global T=0.05 Adapt=0.2": ["73_MG_02_005"]}' --name MG3Seeds --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Generative Replay 2 Samples, γ=0.4": ["73_MS_2_2_04"], "Generative Replay 4 Samples, γ=0.8": ["73_MS_4_05_08"]}' --name MS3Seeds --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"]}' --name FLvsCFCD3Seeds --gen_all_seeds

# python plot/plot_with_drift.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "CF Solo": ["73_CF_Solo"]}' --name CFSolo --gen_all_seeds

# python plot/plot_with_drift.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "CF Solo": ["73_CF_Solo"], "Medium Brightness CF CD": ["MediumBrightnessCFCD"] }' --name MediumBrightness --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Generative Replay 2 Samples, γ=0.4": ["73_MS_2_2_04"], "Combined Method": ["73_MS204_ML02s2_MG02t05"]}' --name MSMLMG3SeedsvsMS --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "60% - 40%": ["art_CFCD_no_weighted_training_6_4"], "70% - 30%": ["art_CFCD_no_weighted_training_7_3"], "80% - 20%": ["art_CFCD_no_weighted_training"]}' --name 64-73-82_3Seeds --gen_all_seeds

# python plot/plot_only_test.py --groups '{"CFCDID": ["art_CFCDID_no_weighted_training_sp1_s3", "art_CFCDID_no_weighted_training_sp2_s3","art_CFCDID_no_weighted_training_sp3_s3","art_CFCDID_no_weighted_training_sp4_s3","art_CFCDID_no_weighted_training_sp5_s3"], "80% - 20%": ["art_CFCD_no_weighted_training_sp1_s3", "art_CFCD_no_weighted_training_sp2_s3","art_CFCD_no_weighted_training_sp3_s3","art_CFCD_no_weighted_training_sp4_s3","art_CFCD_no_weighted_training_sp5_s3"],  "70% - 30%": ["art_CFCD_no_weighted_training_7_3_sp1_s3", "art_CFCD_no_weighted_training_7_3_sp2_s3","art_CFCD_no_weighted_training_7_3_sp3_s3","art_CFCD_no_weighted_training_7_3_sp4_s3","art_CFCD_no_weighted_training_7_3_sp5_s3"], "60% - 40%": ["art_CFCD_no_weighted_training_6_4_sp1_s3", "art_CFCD_no_weighted_training_6_4_sp2_s3","art_CFCD_no_weighted_training_6_4_sp3_s3","art_CFCD_no_weighted_training_6_4_sp4_s3","art_CFCD_no_weighted_training_6_4_sp5_s3"]}' --name 64-73-82_1Seed

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Generative Replay 2 Samples, γ=0.4": ["73_MS_2_2_04"], "Generative Replay 4 Samples, γ=0.8": ["73_MS_4_05_08"]}' --name MSDrift3Seeds  --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Generative Replay 2 Samples, γ=0.4": ["73_MS_2_2_04"], "Generative Replay 4 Samples, γ=0.8": ["73_MS_4_05_08"]}' --name MSVSRepr --gen_all_seeds

#########Feat vs Repr based##########

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Feat-based Gen Sampling": ["73_MS_2_2_04"],  "Repr-based Gen Sampling": ["repr_MS_2_2_04"]}' --name ReprvsFeatMS --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Feat-based Local Training": ["73_ML_02_str1"],  "Repr-based Local Training": ["repr_ML_02"]}' --name ReprvsFeatML --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Feat-based Global Aggregation": ["73_MG_03_05t"],  "Repr-based Global Aggregation": ["repr_MG_03_05t"]}' --name ReprvsFeatMG --gen_all_seeds

# python plot/plot_only_test.py --groups '{"FL Baseline": ["art_CFCDID_no_weighted_training"], "CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Feat-based Combined Method": ["73_MS204_ML02s2_MG02t05"], "Repr-based Combined Method": ["repr_MS204_ML02s2_MG0205t"]}' --name ReprvsFeatMSMLMG --gen_all_seeds




# python plot/plot_only_test.py --groups '{"Split 1": ["art_CFCD_no_weighted_training_7_3_sp1_s1","art_CFCD_no_weighted_training_7_3_sp1_s2","art_CFCD_no_weighted_training_7_3_sp1_s3"], "Split 2": ["art_CFCD_no_weighted_training_7_3_sp2_s1","art_CFCD_no_weighted_training_7_3_sp2_s2","art_CFCD_no_weighted_training_7_3_sp2_s3"], "Split 3": ["art_CFCD_no_weighted_training_7_3_sp3_s1","art_CFCD_no_weighted_training_7_3_sp3_s2","art_CFCD_no_weighted_training_7_3_sp3_s3"], "Split 4": ["art_CFCD_no_weighted_training_7_3_sp4_s1","art_CFCD_no_weighted_training_7_3_sp4_s2","art_CFCD_no_weighted_training_7_3_sp4_s3"], "Split 5": ["art_CFCD_no_weighted_training_7_3_sp5_s1","art_CFCD_no_weighted_training_7_3_sp5_s2","art_CFCD_no_weighted_training_7_3_sp5_s3"]}' --name Baseline_Splits

#  python plot/plot_only_test.py --groups '{"CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Combined Method": ["73_MS204_ML02s2_MG02t05"], "FedProx µ=1": ["fedprox"]}' --name FedProx --gen_all_seeds

#  python plot/plot_only_test.py --groups '{"CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Combined Method": ["73_MS204_ML02s2_MG02t05"], "FedProx µ=0.001": ["FedProx001"],"FedProx µ=0.005": ["FedProx005"],"FedProx µ=0.01": ["FedProx01"],"FedProx µ=1": ["fedprox"],"FedProx µ=10": ["FedProx10"]}' --name FedProxAll --gen_all_seeds

# python plot/plot_only_test.py --groups '{"CF CD Baseline": ["art_CFCD_no_weighted_training_7_3"], "Combined Method": ["73_MS204_ML02s2_MG02t05"], "FedAvgM lr=0.001 β=0.5": ["FedAvgM00105"], "FedAvgM lr=0.1 β=0.5": ["FedAvgM0105"], "FedAvgM lr=0.5 β=0.9": ["FedAvgM"], "FedAvgM lr=0.05 β=0.6": ["FedAvgM00506"]}' --name FedAvgMAll --gen_all_seeds

# python plot/plot_clean_comparison.py --groups '{"Federated Learning": ["1_stage_FL_sp1_s1", "1_stage_FL_sp2_s1","1_stage_FL_sp3_s1","1_stage_FL_sp4_s1","1_stage_FL_sp5_s1","1_stage_FL_sp1_s2", "1_stage_FL_sp2_s2","1_stage_FL_sp3_s2","1_stage_FL_sp4_s2","1_stage_FL_sp5_s2","1_stage_FL_sp1_s3", "1_stage_FL_sp2_s3","1_stage_FL_sp3_s3","1_stage_FL_sp4_s3","1_stage_FL_sp5_s3"], "Centralised": ["1_stage_Centralized_sp1_s1", "1_stage_Centralized_sp2_s1","1_stage_Centralized_sp3_s1","1_stage_Centralized_sp4_s1","1_stage_Centralized_sp5_s1","1_stage_Centralized_sp1_s2", "1_stage_Centralized_sp2_s2","1_stage_Centralized_sp3_s2","1_stage_Centralized_sp4_s2","1_stage_Centralized_sp5_s2","1_stage_Centralized_sp1_s3", "1_stage_Centralized_sp2_s3","1_stage_Centralized_sp3_s3","1_stage_Centralized_sp4_s3","1_stage_Centralized_sp5_s3"]}' --name FLvsCentralized_clean_3