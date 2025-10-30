"""
Cross-validation comparison plotting for multiple federated learning experiments.

This module provides functionality to compare cross-validation results across multiple experiments.
"""

import matplotlib.pyplot as plt
from plot_utils import (get_available_folds, filter_metrics, create_figure, 
                        calculate_global_parameters, hide_empty_subplots, 
                        save_plot, create_mode_description, create_filename_suffix)
from statistics import compute_fold_statistics
from plotting import plot_metric, create_legend_subplot


def plot_cross_validation_comparison(experiment_names: list, submodel: str = 'MM', metric_filter: str = 'test',
                                    show_individual_clients: bool = False, show_full_training: bool = False,
                                    smooth_window: int = 0, show_std: bool = True, folds: list = None):
    """
    Compare cross-validation results from multiple federated learning experiments
    
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
    metrics_to_plot = filter_metrics(metric_filter)
    
    # Create plot
    fig, axs = create_figure(len(metrics_to_plot))
    
    # Create title
    mode_desc = create_mode_description(show_individual_clients, show_full_training, smooth_window, show_std)
    exp_names_str = " vs ".join(experiment_names)
    fig.suptitle(f"Cross-Validation Comparison: {exp_names_str}\n"
                f"({submodel} - {metric_filter} metrics, {mode_desc})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Calculate global parameters from all experiments and folds
    global_all_rounds, global_max_steps_per_round = calculate_global_parameters(
        None, all_experiment_data
    )
    
    # Compute statistics for all experiments
    all_experiment_stats = _compute_all_experiment_statistics(
        all_experiment_data, metrics_to_plot, submodel, global_all_rounds, 
        global_max_steps_per_round, show_full_training, show_individual_clients, smooth_window
    )
    
    # Setup for plotting
    ax_dims = axs.shape if hasattr(axs, 'shape') else (1, 1)
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    has_legend_entry = 'Legend' in metrics_to_plot
    
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
        
        # Plot each experiment's cross-validation results
        for exp_idx, (exp_name, fold_numbers) in enumerate(all_experiment_data.items()):
            if exp_name not in all_experiment_stats or metric not in all_experiment_stats[exp_name]:
                continue
                
            client_stats, server_stats = all_experiment_stats[exp_name][metric]
            color = exp_colors[exp_idx % len(exp_colors)]
            
            plot_metric(ax, metric, title, client_stats, server_stats, global_all_rounds,
                       global_max_steps_per_round, show_full_training, show_individual_clients,
                       show_std, show_legend=not has_legend_entry, color=color, 
                       label_prefix=exp_name)
    
    # Create comparison legend if needed
    if has_legend_entry:
        comparison_legend_handles = _create_comparison_legend_handles(
            all_experiment_data.keys(), exp_colors
        )
        create_legend_subplot(axs, ax_dims, ax_width, metrics_to_plot, 
                             legend_handles=comparison_legend_handles)
    
    # Hide empty subplots
    total_plots = len(metrics_to_plot)
    hide_empty_subplots(axs, total_plots)
    
    plt.tight_layout()
    
    # Save the figure
    suffix = create_filename_suffix(show_individual_clients, show_full_training, smooth_window, show_std)
    exp_names_clean = "_vs_".join([exp.replace("_", "-") for exp in experiment_names])
    filename = f"crossval_comparison_{exp_names_clean}_{submodel.lower()}_{metric_filter}_{suffix}.png"
    plot_path = save_plot(fig, filename)
    
    plt.show()
    return plot_path


def _compute_all_experiment_statistics(all_experiment_data, metrics_to_plot, submodel, 
                                      global_all_rounds, global_max_steps_per_round,
                                      show_full_training, show_individual_clients, smooth_window):
    """Compute statistics for all experiments"""
    all_experiment_stats = {}
    
    for exp_name, fold_numbers in all_experiment_data.items():
        print(f"Computing cross-validation statistics for: {exp_name}")
        all_experiment_stats[exp_name] = {}
        
        for base_metric in metrics_to_plot.keys():
            if base_metric == 'Legend':
                continue
            metric = f"{base_metric}/{submodel}"
            
            client_stats, server_stats = compute_fold_statistics(
                exp_name, fold_numbers, metric, global_all_rounds, global_max_steps_per_round,
                show_full_training, show_individual_clients, smooth_window
            )
            
            all_experiment_stats[exp_name][metric] = (client_stats, server_stats)
    
    return all_experiment_stats


def _create_comparison_legend_handles(experiment_names, exp_colors):
    """Create legend handles for comparison plots"""
    comparison_legend_handles = []
    for exp_idx, exp_name in enumerate(experiment_names):
        color = exp_colors[exp_idx % len(exp_colors)]
        line = plt.Line2D([0], [0], color=color, linewidth=2, label=f'{exp_name} (CV avg)')
        comparison_legend_handles.append(line)
    
    return comparison_legend_handles