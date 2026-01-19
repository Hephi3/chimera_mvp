"""
Common utilities for cross-validation plotting of federated learning experiments.

This module contains shared functions for data loading, processing, and visualization
that are used by both single experiment and comparison plotting scripts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict


# Configuration
ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cdcf/results"

# Federated learning metrics - base metrics without submodel specification
FEDERATED_METRICS = {
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

# Visual styling
LINE_STYLES = ['-', '--', '-.', ':']
SUBMODELS = {
    'CLAM': LINE_STYLES[1],
    'CD': LINE_STYLES[2],
    'MM': LINE_STYLES[0]
}

# Alpha (transparency) settings
LINE_ALPHA = 0.8
STD_ALPHA = 0.3


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


def tensorboard_to_datadict_federated(experiment_name: str, fold_num: int, exp_dir: str = ROOT_RESULTS):
    """Extract data from TensorBoard logs for federated learning plotting from a specific fold"""
    
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


def create_figure(num_metrics=None):
    """Create figure for federated learning plots"""
    if num_metrics is None:
        num_metrics = len(FEDERATED_METRICS)
    
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


def get_plot_rounds(metric, all_rounds):
    """Get the rounds to plot based on metric type"""
    # For train/val metrics, exclude round 0 (initial evaluation before training)
    if 'test' not in metric:
        return [r for r in all_rounds if r > 0]
    else:
        return all_rounds


def get_client_colors():
    """Get standard colors for individual clients"""
    return ['blue', 'red', 'green', 'orange', 'purple', 'brown']


def get_client_line_styles():
    """Get standard line styles for individual clients"""
    return ['-', '--', '-.', ':']


def filter_metrics(metric_filter):
    """Filter metrics based on metric_filter parameter"""
    if metric_filter == 'all':
        return {k: v for k, v in FEDERATED_METRICS.items() 
                if k is not None and (v is not None or k == 'Legend')}
    else:
        return {k: v for k, v in FEDERATED_METRICS.items() 
                if k is not None and (v is not None or k == 'Legend') and 
                (metric_filter in k or k == 'Legend')}


def calculate_global_parameters(experiment_data, all_experiment_data=None, experiment_name=None):
    """
    Calculate global parameters from experiment data.
    
    Args:
        experiment_data: For single experiment - list of fold numbers
                        For comparison - dict {exp_name: fold_numbers}
        all_experiment_data: For comparison mode - pass the same as experiment_data
        experiment_name: For single experiment mode - name of the experiment
    
    Returns:
        Tuple of (global_all_rounds, global_max_steps_per_round)
    """
    global_all_rounds = set()
    global_max_steps_per_round = 0
    
    # Handle both single experiment and comparison modes
    if all_experiment_data is None:
        # Single experiment mode
        fold_numbers = experiment_data
        experiments_to_process = [(experiment_name, fold_numbers)]
    else:
        # Comparison mode
        experiments_to_process = all_experiment_data.items()
    
    for exp_name, fold_numbers in experiments_to_process:
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
    
    return sorted(global_all_rounds), global_max_steps_per_round


def hide_empty_subplots(axs, total_plots):
    """Hide empty subplots that aren't being used"""
    if hasattr(axs, 'flat'):
        for j in range(total_plots, len(axs.flat)):
            axs.flat[j].axis('off')


def save_plot(fig, filename, exp_dir=ROOT_RESULTS):
    """Save plot to file"""
    plot_path = os.path.join(exp_dir, filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    return plot_path


def create_mode_description(show_individual_clients, show_full_training, smooth_window, show_std):
    """Create description of plotting mode for titles"""
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
    
    return ", ".join(mode_desc)


def create_filename_suffix(show_individual_clients, show_full_training, smooth_window, show_std):
    """Create filename suffix based on plotting options"""
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
    
    return "_".join(suffix_parts)