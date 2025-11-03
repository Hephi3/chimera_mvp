"""
Plotting functions for cross-validation analysis of federated learning experiments.

This module contains functions for plotting individual metrics and creating legends.
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (LINE_ALPHA, STD_ALPHA, get_plot_rounds, get_client_colors, 
                        get_client_line_styles)


def plot_metric(ax, metric, title, client_stats, server_stats, all_rounds, 
               global_max_steps_per_round, show_full_training, show_individual_clients, 
               show_std=True, show_legend=True, color='blue', label_prefix=''):
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
        color: Color for the plot lines (used in comparison mode)
        label_prefix: Prefix for labels (used in comparison mode)
    """
    
    plot_rounds = get_plot_rounds(metric, all_rounds)
    
    if 'test' in metric:
        _plot_test_metric(ax, metric, client_stats, server_stats, plot_rounds, 
                         show_individual_clients, show_std, color, label_prefix)
    else:
        _plot_train_val_metric(ax, metric, client_stats, server_stats, plot_rounds,
                              global_max_steps_per_round, show_full_training, 
                              show_individual_clients, show_std, color, label_prefix)
    
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)


def _plot_test_metric(ax, metric, client_stats, server_stats, plot_rounds, 
                     show_individual_clients, show_std, color, label_prefix):
    """Plot test metrics (server evaluation and optionally individual clients)"""
    
    if show_individual_clients and client_stats:
        # Plot individual client test statistics
        client_colors = get_client_colors()
        
        for client_idx, client_id in enumerate(sorted(client_stats.keys())):
            client_color = client_colors[client_idx % len(client_colors)] if not label_prefix else color
            
            client_means, client_stds, client_rounds = _extract_client_values(
                client_stats, client_id, plot_rounds
            )
            
            if client_means:
                # Plot client mean line with dashed style
                client_label = f'{label_prefix} C{client_id} (CV avg)' if label_prefix else f'Client {client_id} (CV avg)'
                ax.plot(client_rounds, client_means, color=client_color, marker='o', linestyle='--',
                       linewidth=2, markersize=4, label=client_label, alpha=LINE_ALPHA)
                
                # Plot standard deviation shaded band for individual clients
                if show_std and client_stds and any(std > 0 for std in client_stds):
                    client_means_array = np.array(client_means)
                    client_stds_array = np.array(client_stds)
                    ax.fill_between(client_rounds, 
                                   client_means_array - client_stds_array, 
                                   client_means_array + client_stds_array,
                                   color=client_color, alpha=STD_ALPHA * 0.6)
    
    # Always show server evaluation for test metrics
    if server_stats:
        server_means, server_stds, server_rounds = _extract_server_values(server_stats, plot_rounds)
        
        if server_means:
            # Plot mean line - use square markers to distinguish from clients when showing individual clients
            server_color = 'black' if (show_individual_clients and client_stats and not label_prefix) else color
            server_marker = 's' if show_individual_clients and client_stats else 'o'
            server_label_suffix = ' (Global)' if show_individual_clients and client_stats else ' (CV avg)'
            label = f'{label_prefix}{server_label_suffix}' if label_prefix else 'Cross-Val Average'
            
            ax.plot(server_rounds, server_means, color=server_color, marker=server_marker, 
                   linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA)
            
            # Plot standard deviation band
            if show_std and any(std > 0 for std in server_stds):
                server_means_array = np.array(server_means)
                server_stds_array = np.array(server_stds)
                ax.fill_between(server_rounds, 
                               server_means_array - server_stds_array, 
                               server_means_array + server_stds_array,
                               color=server_color, alpha=STD_ALPHA)
            
            ax.set_xlabel('Federated Round')


def _plot_train_val_metric(ax, metric, client_stats, server_stats, plot_rounds,
                          global_max_steps_per_round, show_full_training, 
                          show_individual_clients, show_std, color, label_prefix):
    """Plot training/validation metrics"""
    
    if not plot_rounds or not client_stats:
        return
    
    if show_individual_clients:
        _plot_individual_clients(ax, client_stats, plot_rounds, global_max_steps_per_round,
                               show_full_training, show_std, color, label_prefix)
    else:
        _plot_aggregated_clients(ax, client_stats, plot_rounds, global_max_steps_per_round,
                               show_full_training, show_std, color, label_prefix)
    
    # Plot server evaluation statistics if available for train/val metrics
    if server_stats:
        _plot_server_evaluation(ax, server_stats, plot_rounds, show_std, color, label_prefix)


def _plot_individual_clients(ax, client_stats, plot_rounds, global_max_steps_per_round,
                           show_full_training, show_std, color, label_prefix):
    """Plot individual client statistics"""
    client_colors = get_client_colors()
    client_line_styles = get_client_line_styles()
    
    for client_idx, client_id in enumerate(sorted(client_stats.keys())):
        client_color = client_colors[client_idx % len(client_colors)] if not label_prefix else color
        client_line_style = client_line_styles[client_idx % len(client_line_styles)] if not label_prefix else '-'
        
        if show_full_training:
            _plot_client_full_training(ax, client_stats, client_id, plot_rounds, 
                                     global_max_steps_per_round, show_std, 
                                     client_color, client_line_style, label_prefix)
        else:
            _plot_client_final_values(ax, client_stats, client_id, plot_rounds, 
                                    show_std, client_color, client_line_style, label_prefix)


def _plot_aggregated_clients(ax, client_stats, plot_rounds, global_max_steps_per_round,
                           show_full_training, show_std, color, label_prefix):
    """Plot aggregated statistics across all clients"""
    if show_full_training:
        _plot_aggregated_full_training(ax, client_stats, plot_rounds, global_max_steps_per_round,
                                     show_std, color, label_prefix)
    else:
        _plot_aggregated_final_values(ax, client_stats, plot_rounds, show_std, color, label_prefix)


def _extract_client_values(client_stats, client_id, plot_rounds):
    """Extract client values for plotting"""
    client_means = []
    client_stds = []
    client_rounds = []
    
    for round_num in sorted(client_stats[client_id].keys()):
        if round_num in plot_rounds:
            mean_val = client_stats[client_id][round_num]['mean']
            std_val = client_stats[client_id][round_num]['std']
            
            # Handle both single values and lists
            if isinstance(mean_val, list):
                mean_val = mean_val[-1] if mean_val else 0.0
            if isinstance(std_val, list):
                std_val = std_val[-1] if std_val else 0.0
                
            client_means.append(mean_val)
            client_stds.append(std_val)
            client_rounds.append(round_num)
    
    return client_means, client_stds, client_rounds


def _extract_server_values(server_stats, plot_rounds):
    """Extract server values for plotting"""
    server_means = []
    server_stds = []
    server_rounds = []
    
    for round_num in sorted(server_stats.keys()):
        if round_num in plot_rounds:
            server_means.append(server_stats[round_num]['mean'])
            server_stds.append(server_stats[round_num]['std'])
            server_rounds.append(round_num)
    
    return server_means, server_stds, server_rounds


def _plot_client_full_training(ax, client_stats, client_id, plot_rounds, 
                              global_max_steps_per_round, show_std, 
                              client_color, client_line_style, label_prefix):
    """Plot full training curves for a client"""
    all_means = []
    all_stds = []
    all_steps = []
    
    for round_idx, round_num in enumerate(plot_rounds):
        if round_num in client_stats[client_id]:
            round_means = client_stats[client_id][round_num]['mean']
            round_stds = client_stats[client_id][round_num]['std']
            
            # Create x-positions: each round starts at round_idx * global_max_steps_per_round
            round_start = round_idx * global_max_steps_per_round
            round_steps = list(range(round_start, round_start + len(round_means)))
            
            all_means.extend(round_means)
            all_stds.extend(round_stds)
            all_steps.extend(round_steps)
    
    if all_means:
        # Plot mean line
        label = f'{label_prefix} C{client_id} (CV avg)' if label_prefix else f'Client {client_id} (CV avg)'
        ax.plot(all_steps, all_means, color=client_color, linewidth=2, linestyle=client_line_style,
               label=label, alpha=LINE_ALPHA)
        
        # Plot standard deviation band
        if show_std and any(std > 0 for std in all_stds):
            all_means_array = np.array(all_means)
            all_stds_array = np.array(all_stds)
            ax.fill_between(all_steps, 
                           all_means_array - all_stds_array, 
                           all_means_array + all_stds_array,
                           color=client_color, alpha=STD_ALPHA)


def _plot_client_final_values(ax, client_stats, client_id, plot_rounds, 
                             show_std, client_color, client_line_style, label_prefix):
    """Plot final values only for a client"""
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
        ax.plot(round_positions, round_means, color=client_color, marker='o', linestyle=client_line_style,
               linewidth=2, markersize=6, label=label, alpha=LINE_ALPHA)
        
        # Plot standard deviation error bars
        if show_std and any(std > 0 for std in round_stds):
            ax.errorbar(round_positions, round_means, yerr=round_stds,
                       color=client_color, alpha=STD_ALPHA, capsize=5)


def _plot_aggregated_full_training(ax, client_stats, plot_rounds, global_max_steps_per_round,
                                 show_std, color, label_prefix):
    """Plot aggregated full training curves"""
    all_means = []
    all_stds = []
    all_steps = []
    
    for round_idx, round_num in enumerate(plot_rounds):
        if round_num in client_stats:
            round_means = client_stats[round_num]['mean']
            round_stds = client_stats[round_num]['std']
            
            # Create x-positions: each round starts at round_idx * global_max_steps_per_round
            round_start = round_idx * global_max_steps_per_round
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
            all_means_array = np.array(all_means)
            all_stds_array = np.array(all_stds)
            ax.fill_between(all_steps, 
                           all_means_array - all_stds_array, 
                           all_means_array + all_stds_array,
                           color=color, alpha=STD_ALPHA)
    
    # Add round boundaries
    if len(plot_rounds) > 1:
        for round_idx in range(1, len(plot_rounds)):
            boundary = round_idx * global_max_steps_per_round
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Set x-axis labels
    if plot_rounds and global_max_steps_per_round > 0:
        round_centers = [(i + 0.5) * global_max_steps_per_round for i in range(len(plot_rounds))]
        ax.set_xticks(round_centers)
        ax.set_xticklabels([f'Round {r}' for r in plot_rounds])


def _plot_aggregated_final_values(ax, client_stats, plot_rounds, show_std, color, label_prefix):
    """Plot aggregated final values"""
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


def _plot_server_evaluation(ax, server_stats, plot_rounds, show_std, color, label_prefix):
    """Plot server evaluation statistics"""
    server_means, server_stds, server_rounds = _extract_server_values(server_stats, plot_rounds)
    
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


def create_legend_subplot(axs, ax_dims, ax_width, metrics_to_plot, show_individual_clients=False, 
                         client_ids=None, legend_handles=None):
    """Create a dedicated legend subplot"""
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        if base_metric == 'Legend':
            # Get the subplot for legend
            if len(ax_dims) > 1:
                legend_ax = axs[i // ax_width, i % ax_width]
            else:
                legend_ax = axs[i] if hasattr(axs, '__len__') else axs
            
            legend_ax.axis('off')
            
            if legend_handles:
                # For comparison plots
                legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
                legend_ax.set_title('Cross-Validation Comparison')
            else:
                # For single experiment plots
                handles = _create_single_experiment_legend_handles(show_individual_clients, client_ids)
                legend_ax.legend(handles=handles, loc='center', fontsize=12)
                legend_ax.set_title('Cross-Validation Results')
            break


def _create_single_experiment_legend_handles(show_individual_clients, client_ids):
    """Create legend handles for single experiment plots"""
    legend_handles = []
    
    # Add client color legend if showing individual clients
    if show_individual_clients and client_ids:
        client_colors = get_client_colors()
        
        # Add client legend entries
        for client_idx, client_id in enumerate(sorted(client_ids)):
            client_color = client_colors[client_idx % len(client_colors)]
            # Use dashed line for test metrics, solid for train/val
            line_style = '--'  # Default to dashed for test metrics
            line = plt.Line2D([0], [0], color=client_color, linewidth=2, linestyle=line_style,
                             label=f'Client {client_id}')
            legend_handles.append(line)
    
    # Add server/global model legend
    server_color = 'black' if show_individual_clients and client_ids else 'blue'
    server_marker = 's' if show_individual_clients and client_ids else 'o'
    server_label = 'Global Model (CV avg)' if show_individual_clients and client_ids else 'Cross-Validation Average'
    server_line = plt.Line2D([0], [0], color=server_color, marker=server_marker, 
                           linewidth=2, markersize=6, label=server_label)
    legend_handles.append(server_line)
    
    return legend_handles