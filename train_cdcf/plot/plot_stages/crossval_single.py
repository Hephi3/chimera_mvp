"""
Cross-validation plotting for single federated learning experiments.

This module provides functionality to plot cross-validation results for a single experiment
with multiple folds.
"""

import matplotlib.pyplot as plt
from plot_utils import (get_available_folds, filter_metrics, create_figure, 
                        calculate_global_parameters, hide_empty_subplots, 
                        save_plot, create_mode_description, create_filename_suffix)
from statistics import compute_fold_statistics
from plotting import plot_metric, create_legend_subplot


def plot_cross_validation(experiment_name: str, submodel: str = 'MM', metric_filter: str = 'test',
                         show_individual_clients: bool = False, show_full_training: bool = False,
                         smooth_window: int = 0, show_std: bool = True, folds: list = None):
    """
    Plot cross-validation results for a federated learning experiment
    
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
    metrics_to_plot = filter_metrics(metric_filter)
    
    # Create plot
    fig, axs = create_figure(len(metrics_to_plot))
    
    # Create title
    mode_desc = create_mode_description(show_individual_clients, show_full_training, smooth_window, show_std)
    fig.suptitle(f"Cross-Validation Results: {experiment_name}\n"
                f"({submodel} - {metric_filter} metrics, {len(fold_numbers)} folds, {mode_desc})", 
                fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Calculate global parameters from all folds
    global_all_rounds, global_max_steps_per_round = calculate_global_parameters(
        fold_numbers, experiment_name=experiment_name
    )
    
    # Setup for plotting
    ax_dims = axs.shape if hasattr(axs, 'shape') else (1, 1)
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    has_legend_entry = 'Legend' in metrics_to_plot
    
    # Collect client information for legend if needed
    all_client_ids = set()
    if show_individual_clients and has_legend_entry:
        all_client_ids = _collect_client_ids(experiment_name, fold_numbers)
    
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
        
        # Compute statistics for this metric
        client_stats, server_stats = compute_fold_statistics(
            experiment_name, fold_numbers, metric, global_all_rounds, global_max_steps_per_round,
            show_full_training, show_individual_clients, smooth_window
        )
        
        plot_metric(ax, metric, title, client_stats, server_stats, global_all_rounds,
                   global_max_steps_per_round, show_full_training, show_individual_clients,
                   show_std, show_legend=not has_legend_entry)
    
    # Create centralized legend if needed
    if has_legend_entry:
        create_legend_subplot(axs, ax_dims, ax_width, metrics_to_plot, 
                             show_individual_clients, all_client_ids)
    
    # Hide empty subplots
    total_plots = len(metrics_to_plot)
    hide_empty_subplots(axs, total_plots)
    
    plt.tight_layout()
    
    # Save the figure
    suffix = create_filename_suffix(show_individual_clients, show_full_training, smooth_window, show_std)
    filename = f"crossval_{experiment_name}_{submodel.lower()}_{metric_filter}_{suffix}.png"
    plot_path = save_plot(fig, filename)
    
    plt.show()
    return plot_path


def _collect_client_ids(experiment_name, fold_numbers):
    """Collect all client IDs across folds for legend purposes"""
    from plot_utils import tensorboard_to_datadict_federated
    
    all_client_ids = set()
    for fold_num in fold_numbers:
        try:
            client_data, _ = tensorboard_to_datadict_federated(experiment_name, fold_num)
            all_client_ids.update(client_data.keys())
        except Exception as e:
            print(f"Warning: Could not load data for fold {fold_num}: {e}")
            continue
    
    return all_client_ids