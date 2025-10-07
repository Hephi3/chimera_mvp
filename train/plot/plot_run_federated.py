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
    # 'ROC_AUC/train': 'Training ROC AUC',
    # 'Accuracy/train': 'Training Accuracy',
    
    'Accuracy/val': 'Validation Accuracy',

    # 'Binary_Accuracy/val': 'Validation Binary Accuracy',
    'F1/val': 'Validation F1 Score',
    'Loss/val': 'Validation Loss',
    # 'ROC_AUC/val': 'Validation ROC AUC',
    
    
    'Binary_Accuracy/test': 'Test Binary Accuracy',
    # 'ROC_AUC/test': 'Test ROC AUC',
    'F1/test': 'Test F1 Score',
    # 'Accuracy/test': 'Test Accuracy',
    'Legend': None,
}

# federated_metrics = {
#     'Loss/train': 'Training Loss',
#     'Binary_Accuracy/train': 'Training Binary Accuracy',
#     'F1/train': 'Training F1 Score',
#     # 'ROC_AUC/train': 'Training ROC AUC',
#     # 'Accuracy/train': 'Training Accuracy',
    
#     'Loss/val': 'Validation Loss',
#     'Binary_Accuracy/val': 'Validation Binary Accuracy',
#     'F1/val': 'Validation F1 Score',
#     # 'ROC_AUC/val': 'Validation ROC AUC',
#     # 'Accuracy/val': 'Validation Accuracy',
    
#     'Binary_Accuracy/test': 'Test Binary Accuracy',
#     # 'ROC_AUC/test': 'Test ROC AUC',
#     'F1/test': 'Test F1 Score',
#     # 'Accuracy/test': 'Test Accuracy',
#     'Legend': None,
# }

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
    client_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {client_id: {round: {metric: [values]}}}
    server_data = defaultdict(lambda: defaultdict(dict))  # {round: {metric: value}}
    
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


def plot_federated_metric(ax, metric, title, client_data, server_data, client_colors, linestyle='-', show_legend=True):
    """Plot a single metric for federated learning"""
    
    # Get all rounds that exist
    all_rounds = set()
    for client_id in client_data.keys():
        all_rounds.update(client_data[client_id].keys())
    all_rounds = sorted(all_rounds)
    
    # For train/val metrics, exclude round 0 (initial evaluation before training)
    if 'test' not in metric:
        all_rounds = [r for r in all_rounds if r > 0]
    
    max_steps_per_round = 0
    # Find maximum steps per round across all clients
    for client_id in client_data.keys():
        for round_num in all_rounds:
            if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                steps_in_round = len([v for v in client_data[client_id][round_num][metric] if v is not None])
                max_steps_per_round = max(max_steps_per_round, steps_in_round)
    
    # Plot each client and collect all x-coordinates for proper axis limits
    all_x_coords = []
    for client_id in sorted(client_data.keys()):
        color = client_colors[client_id % len(client_colors)]
        client_rounds = client_data[client_id]
        
        all_values = []
        all_steps = []
        
        for round_idx, round_num in enumerate(all_rounds):
            if round_num in client_rounds and metric in client_rounds[round_num]:
                round_values = [v for v in client_rounds[round_num][metric] if v is not None]
                
                # Create x-positions: each round starts at round_idx * max_steps_per_round
                round_start = round_idx * max_steps_per_round
                round_steps = list(range(round_start, round_start + len(round_values)))
                
                all_values.extend(round_values)
                all_steps.extend(round_steps)
        
        if all_values:
            ax.plot(all_steps, all_values, color=color, alpha=0.7, linestyle=linestyle,
                   label=f'Client {client_id}')
            all_x_coords.extend(all_steps)
    
    # Add vertical lines for round boundaries (only if there are multiple rounds)
    if len(all_rounds) > 1:
        round_boundaries = []
        for round_idx in range(1, len(all_rounds)):
            boundary = round_idx * max_steps_per_round
            round_boundaries.append(boundary)
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Plot server evaluation points
    server_values = []
    server_rounds = []
    for round_num in sorted(server_data.keys()):
        if metric in server_data[round_num]:
            server_values.append(server_data[round_num][metric])
            server_rounds.append(round_num)
    
    if server_values:
        # For test metrics, plot server evaluations at their actual round numbers
        if 'test' in metric:
            # Hide left axis since clients don't train on test data
            ax.set_visible(False)
            
            # Use the main axis for server evaluation (cleaner than dual axes)
            ax.scatter(server_rounds, server_values, color='red', marker='s', 
                      s=100, alpha=0.9, label='Global Model (Server)', zorder=5, edgecolor='black')
            ax.plot(server_rounds, server_values, color='red', alpha=0.6, 
                   linestyle=':', linewidth=2, zorder=4)
            
            # Set x-axis to show round numbers
            ax.set_xlim(-0.5, max(server_rounds) + 0.5)
            ax.set_xticks(server_rounds)
            ax.set_xticklabels([f'Round {r}' for r in server_rounds])
            ax.set_xlabel('Federated Round')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_visible(True)  # Make sure the axis is visible
            ax.grid(True, alpha=0.3)
            
            # Add legend for test metrics only if show_legend is True
            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
            
            return  # Exit early for test metrics to avoid the general x-axis handling below
        else:
            # For train/val metrics, position server evaluations at round boundaries
            server_x_positions = []
            for round_num in server_rounds:
                if round_num == 0:
                    server_x_positions.append(0)  # Initial evaluation at x=0
                else:
                    # After round completion: at the end of the round
                    round_idx = round_num - 1  # Convert to 0-based index for completed rounds
                    if round_idx < len(all_rounds):
                        server_x_positions.append((round_idx + 1) * max_steps_per_round)
            
            # Ensure we have matching lengths
            min_len = min(len(server_x_positions), len(server_values))
            server_x_positions = server_x_positions[:min_len]
            server_values = server_values[:min_len]
            
            if len(server_x_positions) > 0:
                ax.scatter(server_x_positions, server_values, color='red', marker='s', 
                          s=100, alpha=0.9, label='Global Model (Server)', zorder=5, edgecolor='black')
                ax.plot(server_x_positions, server_values, color='red', alpha=0.6, 
                       linestyle=':', linewidth=2, zorder=4)
                # Add server x-coordinates to the range calculation
                all_x_coords.extend(server_x_positions)
    
    # Add round labels on x-axis
    if all_rounds and max_steps_per_round > 0:
        round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(all_rounds))]
        ax.set_xticks(round_centers)
        ax.set_xticklabels([f'Round {r}' for r in all_rounds])
        
        # Set x-axis limits based on actual data range to avoid excessive whitespace
        if all_x_coords:
            min_x = min(all_x_coords)
            max_x = max(all_x_coords)
            x_range = max_x - min_x
            padding = max(x_range * 0.05, 0.5)  # 5% padding or minimum 0.5
            ax.set_xlim(min_x - padding, max_x + padding)
        elif len(all_rounds) == 1:
            # Fallback for single round with no client data
            ax.set_xlim(-max_steps_per_round * 0.1, max_steps_per_round * 1.1)
        else:
            # Fallback for multiple rounds
            ax.set_xlim(-max_steps_per_round * 0.1, len(all_rounds) * max_steps_per_round * 1.05)
    
    ax.set_title(title)
    ax.set_xlabel('Federated Learning Rounds')
    ax.set_ylabel(title)
    
    # Improve legend handling - only add legend if there are labeled artists and show_legend is True
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    ax.grid(True, alpha=0.3)


def create_client_legend_handles(client_data, client_colors):
    legend_handles = []
    for client_id in sorted(client_data.keys()):
        color = client_colors[client_id % len(client_colors)]
        line = plt.Line2D([0], [0], color=color, linewidth=2, label=f'Client {client_id} Training', )
        legend_handles.append(line)
    
    # Add global model legend
    server_line = plt.Line2D([0], [0], color='red', marker='s', markersize=8, 
                            linestyle=':', linewidth=2, label='Global Model Performance')
    legend_handles.append(server_line)
    return legend_handles
    

def create_legend_subplot(axs, ax_dims, ax_width, legend_handles):
    for i, (base_metric, title) in enumerate(federated_metrics.items()):
        if base_metric == 'Legend':
            # Get the subplot for legend
            if len(ax_dims) > 1:
                legend_ax = axs[i // ax_width, i % ax_width]
            else:
                legend_ax = axs[i] if hasattr(axs, '__len__') else axs
            
            legend_ax.axis('off')
            legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
            legend_ax.set_title('Legend')
            break

def plot_federated_experiment(experiment_name: str, submodel: str = 'MM', smooth_window: int = None):
    """
    Plot federated learning experiment results
    
    Args:
        experiment_name: Name of the federated experiment
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD', or 'ALL')
        smooth_window: Window size for smoothing (None = no smoothing)
    """
    # Get federated data
    client_data, server_data = tensorboard_to_datadict_federated(experiment_name)
    # print("Client Data:", list(client_data.keys()))
    # print("Client Data:", list(client_data[0].keys()))
    # print("Client Data:", list(client_data[0][3].keys()))
    # print("Example: 'F1/test/MM'", client_data[1][1].get('F1/test/MM', 'Not Found'), client_data[1][2].get('F1/test/MM', 'Not Found'), client_data[1][3].get('F1/test/MM', 'Not Found'))
    # print("Server Data:", list(server_data.keys()))

    # Apply smoothing if requested
    if smooth_window and smooth_window > 1:
        client_data = smooth_client_data(client_data, smooth_window)

    # Create plot
    fig, axs = create_federated_plot()
    smooth_text = f" (smoothed w={smooth_window})" if smooth_window and smooth_window > 1 else ""
    fig.suptitle(f"Federated Learning: {experiment_name}\nClient Training vs Global Model Performance ({submodel}){smooth_text}", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Colors for different clients - extended palette
    client_colors = [
        'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'lime', 'indigo', 'teal', 'maroon', 'navy',
        'darkgreen', 'darkred', 'darkorange', 'darkviolet', 'darkblue', 'darkgoldenrod'
    ]
    
    ax_dims = axs.shape
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    
    # Check if we should suppress individual legends (when 'Legend' entry exists)
    has_legend_entry = 'Legend' in federated_metrics
    
    for i, (base_metric, title) in enumerate(federated_metrics.items()):
        if base_metric is None or title is None:
            continue
        
        # Skip Legend entry - it will be handled separately
        if base_metric == 'Legend':
            continue
            
        # Get the subplot
        if len(ax_dims) > 1:
            ax = axs[i // ax_width, i % ax_width]
        else:
            ax = axs[i] if hasattr(axs, '__len__') else axs
        
        # Construct full metric name with submodel
        if submodel == 'ALL':
            # Plot all submodels with different line styles
            for sub in ['MM', 'CLAM', 'CD']:
                metric = f"{base_metric}/{sub}"
                plot_federated_metric(ax, metric, title + f" ({sub})", client_data, server_data, 
                                    client_colors, submodels.get(sub, '-'), show_legend=not has_legend_entry)
        else:
            metric = f"{base_metric}/{submodel}"
            plot_federated_metric(ax, metric, title, client_data, server_data, client_colors, 
                                show_legend=not has_legend_entry)
    
    # Find the Legend subplot and create legend there
    legend_handles = create_client_legend_handles(client_data, client_colors)
    create_legend_subplot(axs, ax_dims, ax_width, legend_handles)

    # Hide any remaining empty subplots
    if hasattr(axs, 'flat'):
        for j in range(len(federated_metrics), len(axs.flat)):
            axs.flat[j].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    plot_dir = os.path.join(ROOT_RESULTS, experiment_name)
    plot_path = os.path.join(plot_dir, f"federated_plot_{submodel.lower()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


def create_compare_legend_handles(all_data, exp_colors, client_line_styles, metric_filter, show_individual_clients):
    legend_handles = []
            
    # Add experiment lines
    for exp_idx, exp_name in enumerate(all_data.keys()):
        color = exp_colors[exp_idx % len(exp_colors)]
        
        if 'test' in metric_filter or metric_filter == 'all':
            # Test metrics line
            test_line = plt.Line2D([0], [0], color=color, marker='o', linewidth=2, 
                                    markersize=6, label=f'{exp_name}')
            legend_handles.append(test_line)
        else:
            # Train/val metrics
            if show_individual_clients:
                # Show client examples
                for client_idx in range(min(3, len(client_line_styles))):  # Show max 3 client examples
                    client_line_style = client_line_styles[client_idx % len(client_line_styles)]
                    client_line = plt.Line2D([0], [0], color=color, linestyle=client_line_style, 
                                            linewidth=1.5, 
                                            label=f'{exp_name} C{client_idx}' if client_idx == 0 else '')
                    if client_idx == 0:  # Only add label for first client to avoid clutter
                        legend_handles.append(client_line)
            else:
                # Averaged client line
                avg_line = plt.Line2D([0], [0], color=color, marker='o', linewidth=2, 
                                    markersize=6, label=f'{exp_name} (avg)')
                legend_handles.append(avg_line)
            
            # Server evaluation line
            server_line = plt.Line2D([0], [0], color=color, marker='s', linewidth=2, 
                                    markersize=4, linestyle='--',
                                    label=f'{exp_name} (server)')
            legend_handles.append(server_line)
    return legend_handles

def plot_federated_comparison(experiment_names: list, submodel: str = 'MM', metric_filter: str = 'test', 
                             show_individual_clients: bool = False, show_full_training: bool = False, 
                             smooth_window: int = 0):
    """
    Compare multiple federated learning experiments
    
    Args:
        experiment_names: List of federated experiment names to compare
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD')
        metric_filter: Which type of metrics to plot ('train', 'val', 'test', or 'all')
        show_individual_clients: If True, show individual client curves instead of averages
        show_full_training: If True, show complete training progress within rounds instead of just final values
        smooth_window: If > 0, apply moving average smoothing with this window size
    """
    
    # Alpha (transparency) settings for plot lines - adjust these to change line transparency
    CLIENT_LINE_ALPHA = 0.6      # Alpha for individual client lines
    SERVER_LINE_ALPHA = 0.6      # Alpha for server evaluation lines
    AVERAGED_LINE_ALPHA = 0.6    # Alpha for averaged client curves
    TEST_LINE_ALPHA = 0.6        # Alpha for test metric lines
    
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
        mode_desc.append("averaged clients")
    if show_full_training:
        mode_desc.append("full training curves")
    else:
        mode_desc.append("final values")
    if smooth_window > 0:
        mode_desc.append(f"smoothed (w={smooth_window})")
    
    title_suffix = ", ".join(mode_desc)
    fig.suptitle(f"Federated Learning Comparison ({submodel} - {metric_filter} metrics, {title_suffix})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Colors for different experiments - extended palette for many experiments
    exp_colors = [
        'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'lime', 'indigo', 'teal', 'maroon', 'navy',
        'darkgreen', 'darkred', 'darkorange', 'darkviolet', 'darkblue', 'darkgoldenrod',
        'crimson', 'forestgreen', 'royalblue', 'chocolate', 'steelblue', 'darkslategray',
        'mediumvioletred', 'seagreen', 'slateblue', 'firebrick', 'darkcyan', 'darkmagenta',
        'saddlebrown', 'midnightblue', 'darkslateblue', 'darkturquoise', 'indianred', 'cadetblue'
    ]
    # Line styles for different clients within an experiment
    client_line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
    
    # Get data for all experiments
    all_data = {}
    for exp_name in experiment_names:
        try:
            client_data, server_data = tensorboard_to_datadict_federated(exp_name)
            
            # Apply smoothing if requested
            if smooth_window > 0:
                client_data = smooth_client_data(client_data, smooth_window)
            
            all_data[exp_name] = (client_data, server_data)
        except Exception as e:
            print(f"Warning: Could not load data for {exp_name}: {e}")
            continue
    
    ax_dims = axs.shape if hasattr(axs, 'shape') else (1, 1)
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    
    # Calculate global set of all rounds across ALL experiments for consistent x-axis
    global_all_rounds = set()
    global_max_steps_per_round = 0  # Global max steps for consistent spacing
    for exp_name, (client_data, server_data) in all_data.items():
        for client_id in client_data.keys():
            global_all_rounds.update(client_data[client_id].keys())
            # Calculate max steps per round across all experiments
            for round_num in client_data[client_id].keys():
                for metric_key in client_data[client_id][round_num].keys():
                    if client_data[client_id][round_num][metric_key]:
                        steps_in_round = len([v for v in client_data[client_id][round_num][metric_key] if v is not None])
                        global_max_steps_per_round = max(global_max_steps_per_round, steps_in_round)
        global_all_rounds.update(server_data.keys())
    global_all_rounds = sorted(global_all_rounds)
    
    # For train/val metrics, exclude round 0 (initial evaluation before training)
    global_train_val_rounds = [r for r in global_all_rounds if r > 0]
    
    # Check if we have a 'Legend' entry in the original federated_metrics
    has_legend_entry = 'Legend' in federated_metrics
    
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        # Handle Legend entry specially
        if base_metric == 'Legend':
            continue  # Will handle legend after plotting all metrics
            
        # Get the subplot for regular metrics
        if len(ax_dims) > 1:
            ax = axs[i // ax_width, i % ax_width]
        else:
            ax = axs[i] if hasattr(axs, '__len__') else axs
        
        metric = f"{base_metric}/{submodel}"
        
        # Plot each experiment
        for exp_idx, (exp_name, (client_data, server_data)) in enumerate(all_data.items()):
            color = exp_colors[exp_idx % len(exp_colors)]
            
            if 'test' in base_metric:
                # For test metrics, plot server evaluation points only
                server_values = []
                server_rounds = []
                for round_num in sorted(server_data.keys()):
                    if metric in server_data[round_num]:
                        server_values.append(server_data[round_num][metric])
                        server_rounds.append(round_num)
                
                if server_values:
                    ax.plot(server_rounds, server_values, color=color, marker='o', 
                           linewidth=2, markersize=6, label=exp_name, alpha=TEST_LINE_ALPHA)
                    ax.set_xlabel('Federated Round')
            
            else:
                # For train/val metrics, plot client training curves
                # Use global rounds for consistent x-axis positioning across experiments
                # Exclude round 0 for train/val metrics (round 0 is initial evaluation before training)
                all_rounds = global_train_val_rounds
                
                if not all_rounds:
                    continue
                
                if show_individual_clients:
                    # Plot each client individually with different line styles
                    for client_idx, client_id in enumerate(sorted(client_data.keys())):
                        client_line_style = client_line_styles[client_idx % len(client_line_styles)]
                        
                        if show_full_training:
                            # Show complete training progress within each round
                            all_values = []
                            all_steps = []
                            
                            # Use global max steps per round for consistent spacing across experiments
                            max_steps_per_round = global_max_steps_per_round
                            
                            for round_idx, round_num in enumerate(all_rounds):
                                if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                    round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                    
                                    if round_values:
                                        # Create x-positions: each round starts at round_idx * max_steps_per_round
                                        round_start = round_idx * max_steps_per_round
                                        round_steps = list(range(round_start, round_start + len(round_values)))
                                        
                                        all_values.extend(round_values)
                                        all_steps.extend(round_steps)
                            
                            if all_values:
                                ax.plot(all_steps, all_values, color=color, 
                                       linestyle=client_line_style, linewidth=1.5, alpha=CLIENT_LINE_ALPHA, 
                                       label=f'{exp_name} C{client_id}')
                            
                            # Add round boundaries for clarity (only if multiple rounds)
                            if len(all_rounds) > 1:
                                for round_idx in range(1, len(all_rounds)):
                                    boundary = round_idx * max_steps_per_round
                                    ax.axvline(x=boundary, color=color, linestyle=':', alpha=0.3)
                            
                            # Set x-axis labels to show rounds
                            if all_rounds and max_steps_per_round > 0:
                                round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(all_rounds))]
                                ax.set_xticks(round_centers)
                                ax.set_xticklabels([f'R{r}' for r in all_rounds])
                                ax.set_xlabel('Federated Learning Progress')
                                
                                # Adjust x-axis limits based on actual number of rounds
                                if len(all_rounds) == 1:
                                    ax.set_xlim(-max_steps_per_round * 0.1, max_steps_per_round * 1.1)
                                else:
                                    ax.set_xlim(-max_steps_per_round * 0.1, len(all_rounds) * max_steps_per_round * 1.05)
                        
                        else:
                            # Show only final values per round
                            round_values = []
                            round_x_positions = []
                            
                            for round_num in all_rounds:
                                if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                    client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                    if client_round_values:
                                        round_values.append(client_round_values[-1])  # Take final value of the round
                                        round_x_positions.append(round_num)
                            
                            if round_values:
                                ax.plot(round_x_positions, round_values, color=color, 
                                       linestyle=client_line_style, linewidth=2, markersize=4, 
                                       alpha=CLIENT_LINE_ALPHA, label=f'{exp_name} C{client_id}')
                                ax.set_xlabel('Federated Round')
                
                else:
                    # Plot aggregated average across clients
                    if show_full_training:
                        # Average the full training curves across clients
                        # Use global max steps per round for consistent spacing across experiments
                        max_steps_per_round = global_max_steps_per_round
                        
                        # Create averaged curves
                        all_avg_values = []
                        all_steps = []
                        
                        for round_idx, round_num in enumerate(all_rounds):
                            # Collect all client values for this round
                            round_client_data = []
                            max_steps_this_round = 0
                            
                            for client_id in sorted(client_data.keys()):
                                if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                    client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                    if client_round_values:
                                        round_client_data.append(client_round_values)
                                        max_steps_this_round = max(max_steps_this_round, len(client_round_values))
                            
                            if round_client_data:
                                # Average across clients for each step in this round
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
                            ax.plot(all_steps, all_avg_values, color=color, linewidth=2, 
                                   alpha=AVERAGED_LINE_ALPHA, label=f'{exp_name} (avg)')
                        
                        # Add round boundaries (only if there are multiple rounds)
                        if len(all_rounds) > 1:
                            for round_idx in range(1, len(all_rounds)):
                                boundary = round_idx * max_steps_per_round
                                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
                        
                        # Set x-axis labels
                        if all_rounds and max_steps_per_round > 0:
                            round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(all_rounds))]
                            ax.set_xticks(round_centers)
                            ax.set_xticklabels([f'R{r}' for r in all_rounds])
                            ax.set_xlabel('Federated Learning Progress')
                            
                            # Adjust x-axis limits based on actual number of rounds
                            if len(all_rounds) == 1:
                                ax.set_xlim(-max_steps_per_round * 0.1, max_steps_per_round * 1.1)
                            else:
                                ax.set_xlim(-max_steps_per_round * 0.1, len(all_rounds) * max_steps_per_round * 1.05)
                    
                    else:
                        # Show only averaged final values per round
                        round_aggregated_values = []
                        round_x_positions = []
                        
                        for round_idx, round_num in enumerate(all_rounds):
                            round_values = []
                            # Collect final values from each client for this round
                            for client_id in sorted(client_data.keys()):
                                if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                                    client_round_values = [v for v in client_data[client_id][round_num][metric] if v is not None]
                                    if client_round_values:
                                        round_values.append(client_round_values[-1])  # Take final value of the round
                            
                            if round_values:
                                # Average across clients for this round
                                avg_value = sum(round_values) / len(round_values)
                                round_aggregated_values.append(avg_value)
                                round_x_positions.append(round_num)
                        
                        if round_aggregated_values:
                            ax.plot(round_x_positions, round_aggregated_values, color=color, marker='o', 
                                   linewidth=2, markersize=6, label=f'{exp_name} (avg)', alpha=AVERAGED_LINE_ALPHA)
                            ax.set_xlabel('Federated Round')
                
                # Also plot server evaluation if available for train/val metrics
                server_values = []
                server_rounds = []
                for round_num in sorted(server_data.keys()):
                    if metric in server_data[round_num]:
                        server_values.append(server_data[round_num][metric])
                        server_rounds.append(round_num)
                
                if server_values:
                    ax.plot(server_rounds, server_values, color=color, marker='s', 
                           linewidth=2, markersize=4, linestyle='--', alpha=SERVER_LINE_ALPHA, 
                           label=f'{exp_name} (server)')
        
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        
        # Add legend only if there's no centralized legend AND there are multiple experiments
        if not has_legend_entry and len(all_data) > 1:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # Adjust legend font size based on number of items and mode
                if show_individual_clients and len(handles) > 15:
                    legend_fontsize = 5
                elif show_individual_clients and len(handles) > 10:
                    legend_fontsize = 6
                else:
                    legend_fontsize = 8
                
                ncols = 3 if show_individual_clients and len(handles) > 10 else 2 if show_individual_clients else 1
                ax.legend(fontsize=legend_fontsize, ncol=ncols)

    legend_handles = create_compare_legend_handles(all_data, exp_colors, client_line_styles, metric_filter, show_individual_clients)
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
        suffix_parts.append("averaged")
    
    if show_full_training:
        suffix_parts.append("fulltraining")
    else:
        suffix_parts.append("finalvals")
    
    if smooth_window > 0:
        suffix_parts.append(f"smooth{smooth_window}")
    
    suffix = "_".join(suffix_parts)
    plot_path = os.path.join(ROOT_RESULTS, f"federated_comparison_{submodel.lower()}_{metric_filter}_{suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Plotting Script for TensorBoard logs')
    parser.add_argument('--name', type=str, default='test2_2_client_s1', help='Name of the federated experiment to plot')
    parser.add_argument('--names', nargs='+', help='Names of federated experiments to compare')
    parser.add_argument('--compare', action='store_true', help='Compare multiple federated experiments')
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD', 'ALL'], 
                       help='Which submodel to plot for federated learning')
    parser.add_argument('--metric_filter', type=str, default='test', choices=['train', 'val', 'test', 'all'],
                       help='Which metrics to show in comparison mode')
    parser.add_argument('--show_individual_clients', action='store_true', 
                       help='Show individual client curves instead of averages in comparison mode')
    parser.add_argument('--show_full_training', action='store_true',
                       help='Show complete training progress within rounds instead of just final values')
    parser.add_argument('--smooth_window', type=int, default=0,
                       help='Apply moving average smoothing with this window size (0 = no smoothing)')
    args = parser.parse_args()

    experiment_name = args.name
    
    # Use federated comparison if requested
    if args.compare and args.names:
        plot_federated_comparison(args.names, args.submodel, args.metric_filter, 
                                args.show_individual_clients, args.show_full_training, 
                                args.smooth_window)
    else:
        # Single federated experiment plotting
        plot_federated_experiment(experiment_name, args.submodel, args.smooth_window)
        
        
#python plot_run.py --compare --names 3_client_seed1 3_client_seedx 3_client_seedy --submodel MM --metric_filter test

#cd /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/plot && conda activate osr && 
#python plot_run_federated.py --name 1_client_redo_s1 --submodel MM

#cd /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/plot && python plot_run_federated.py --compare --names 3_clients_s1 3_clients_1_s1 3_clients_2_s1 3_clients_3_s1 --submodel MM --metric_filter all --show_individual_clients --show_full_training --smooth_window 10

# python plot_run_federated.py --compare --names 3_clients_no_overfit_sp1_4115_s1 3_clients_no_overfit_sp2_4115_s1 3_clients_no_overfit_sp3_4115_s1 3_clients_no_overfit_sp4_4115_s1 3_clients_no_overfit_sp5_4115_s1  --submodel MM --metric_filter all --show_individual_clients --show_full_training --smooth_window 10