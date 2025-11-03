"""
Statistics computation for cross-validation analysis of federated learning experiments.

This module contains functions for computing mean and standard deviation statistics 
across folds for cross-validation analysis.
"""

import numpy as np
from plot_utils import tensorboard_to_datadict_federated, smooth_client_data


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
    server_stats = _compute_server_statistics(fold_data, metric, all_rounds)
    
    # Compute client statistics
    if show_individual_clients:
        client_stats = _compute_individual_client_statistics(
            fold_data, metric, all_rounds, show_full_training, global_max_steps_per_round
        )
    else:
        client_stats = _compute_aggregated_client_statistics(
            fold_data, metric, all_rounds, show_full_training, global_max_steps_per_round
        )
    
    return client_stats, server_stats


def _compute_server_statistics(fold_data, metric, all_rounds):
    """Compute server statistics across folds"""
    server_stats = {}
    for round_num in all_rounds:
        round_values = []
        for _, server_data in fold_data:
            if round_num in server_data and metric in server_data[round_num]:
                round_values.append(server_data[round_num][metric])
        
        if round_values:
            server_stats[round_num] = {
                'mean': np.mean(round_values),
                'std': np.std(round_values, ddof=1) if len(round_values) > 1 else 0.0
            }
    
    return server_stats


def _compute_individual_client_statistics(fold_data, metric, all_rounds, show_full_training, global_max_steps_per_round):
    """Compute statistics per individual client"""
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
                    if client_id in client_data and round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                        round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                        if round_values:
                            fold_curves.append(round_values)
                            max_steps_this_round = max(max_steps_this_round, len(round_values))
                
                if fold_curves and max_steps_this_round > 0:
                    client_stats[client_id][round_num] = _compute_curve_statistics(fold_curves, max_steps_this_round)
        
        else:
            # Statistics for final values only
            for round_num in all_rounds:
                final_values = []
                
                for client_data, _ in fold_data:
                    if client_id in client_data and round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                        round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                        if round_values:
                            final_values.append(round_values[-1])  # Take final value
                
                if final_values:
                    client_stats[client_id][round_num] = {
                        'mean': np.mean(final_values),
                        'std': np.std(final_values, ddof=1) if len(final_values) > 1 else 0.0
                    }
    
    return client_stats


def _compute_aggregated_client_statistics(fold_data, metric, all_rounds, show_full_training, global_max_steps_per_round):
    """Compute aggregated statistics across all clients"""
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
                aggregated_stats[round_num] = _compute_curve_statistics(fold_aggregated, max_steps_this_round)
    
    else:
        # Statistics for aggregated final values
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
                aggregated_stats[round_num] = {
                    'mean': np.mean(fold_aggregated_values),
                    'std': np.std(fold_aggregated_values, ddof=1) if len(fold_aggregated_values) > 1 else 0.0
                }
    
    return aggregated_stats


def _compute_curve_statistics(fold_curves, max_steps_this_round):
    """Compute statistics for training curves across folds"""
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
    
    return {
        'mean': mean_curve,
        'std': std_curve
    }