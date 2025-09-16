import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
from collections import defaultdict
import argparse
import numpy as np

ROOT_RUNS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/runs"
ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/results"
ROOT_DELETE = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/results_delete"

# Original metrics for backward compatibility
metrics = {
    
    # 'Accuracy/train': 'Training Accuracy',
    'Binary_Accuracy/train': 'Training Binary Accuracy',
    # 'ROC_AUC/train': 'Training ROC AUC',
    'F1/train': 'Training F1 Score',
    'Loss/train': 'Training Loss',
    'Accuracy/val': 'Validation Accuracy',
    # 'Binary_Accuracy/val': 'Validation Binary Accuracy',
    # 'ROC_AUC/val': 'Validation ROC AUC',
    'F1/val': 'Validation F1 Score',
    'Loss/val': 'Validation Loss',
    
    # 'Avg_Binary_Accuracy/val': 'Average Validation Binary Accuracy',
    # 'Legend': None,
    # 'Avg_ROC_AUC/val': 'Average Validation ROC AUC',
    # 'Avg_F1/val': 'Average Validation F1 Score',
    # None: None,
    # 'Accuracy/test': 'Test Accuracy',
    'Binary_Accuracy/test': 'Test Binary Accuracy',
    'ROC_AUC/test': 'Test ROC AUC',
    'F1/test': 'Test F1 Score',
    # 'Legend': None,
    # None: None,
}

# Federated learning metrics - base metrics without submodel specification
federated_metrics = {
    'Loss/train': 'Training Loss',
    'Binary_Accuracy/train': 'Training Binary Accuracy',
    'F1/train': 'Training F1 Score',
    'ROC_AUC/train': 'Training ROC AUC',
    'Accuracy/train': 'Training Accuracy',
    
    'Loss/val': 'Validation Loss',
    'Binary_Accuracy/val': 'Validation Binary Accuracy',
    'F1/val': 'Validation F1 Score',
    'ROC_AUC/val': 'Validation ROC AUC',
    'Accuracy/val': 'Validation Accuracy',
    
    'Binary_Accuracy/test': 'Test Binary Accuracy',
    'ROC_AUC/test': 'Test ROC AUC',
    'F1/test': 'Test F1 Score',
    'Accuracy/test': 'Test Accuracy',
    'Legend': None,
}

debug_metrics = {
    'Gradients': 'Gradients',
    'Lossweights': 'Loss Weights'
}
    
# metrics = {
    
#     # 'Accuracy/train': 'Training Accuracy',
#     'Binary_Accuracy/train': 'Training Binary Accuracy',
#     'ROC_AUC/train': 'Training ROC AUC',
#     'F1/train': 'Training F1 Score',
#     'Loss/train': 'Training Loss',
#     # 'Accuracy/val': 'Validation Accuracy',
#     'Binary_Accuracy/val': 'Validation Binary Accuracy',
#     'ROC_AUC/val': 'Validation ROC AUC',
#     'F1/val': 'Validation F1 Score',
#     'Loss/val': 'Validation Loss',
#     'Avg_Binary_Accuracy/val': 'Average Validation Binary Accuracy',
#     'Avg_ROC_AUC/val': 'Average Validation ROC AUC',
#     'Avg_F1/val': 'Average Validation F1 Score',
#     None: None,
#     # 'Accuracy/test': 'Test Accuracy',
#     'Binary_Accuracy/test': 'Test Binary Accuracy',
#     'ROC_AUC/test': 'Test ROC AUC',
#     'F1/test': 'Test F1 Score',
#     'Legend': None,
#     # None: None,
# }
line_styles = ['-', '--', '-.', ':']
submodels = {
    'CLAM': line_styles[1],
    'CD': line_styles[2],
    'MM': line_styles[0]
}



def plot_run(data: dict, smoothed_data: dict, experiment_name: str, config_data:str, show_config:bool = True, axs=None, label=None, test={}, color = 'b', marker = None, avg_datas =None, legend_handles=None, show_phases=False):   
    epochs = len(data[0].keys())
    df = datadict_to_df(data)
    smoothed_df = datadict_to_df(smoothed_data)    
    epochs_per_client = len(data[0][list(data[0].keys())[0]]) if data else 0
    # Get number of experiments out of legend_handles
    
    ax_dims = axs.shape
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    
    def plot_single_line(value, label, color, marker, title, ax, linestyle='-', alpha=0.6):
        # Plot average validation metrics
        ax.axhline(y=value, label=label, alpha=alpha, color=color, linestyle=linestyle)
        
        # Generate random x-positions for markers (between x-axis limits)
        # random_x_positions = np.random.random(1)
        if legend_handles is not None:
            num_experiments = len(legend_handles)
            # Find handle index of the label, if it exists
            pos = next((i for i, h in enumerate(legend_handles) if h.get_label() == label), None)
            if pos is not None:
                random_x_positions = pos/num_experiments * epochs_per_client * len(data)
            else:
                random_x_positions = np.random.random(1)
            
            
        else:
            random_x_positions = np.random.random(1)
        
        
        # Add scatter points at those positions on the horizontal line
        if marker is not None:
            ax.scatter([random_x_positions], [value], 
                    color=color, marker=marker, s=100, alpha=0.8)
        
        ax.set_title(title)
        ax.set_ylabel(title)
        if label and not "Legend" in metrics: ax.legend()
    
    def plot_multisubmodel(df, smoothed_df, metric, title, label, ax):
        for i, submetric in enumerate([f'{metric}/{submodel}' for submodel in list(submodels.keys())]):
            submodel = list(submodels.keys())[i]
            label += f" {submetric}"
            if submetric not in df.columns:
                continue
            if 'test' in submetric:
                ax.plot(df.index, df[submetric], color=color, alpha=0.3, linestyle=submodels[submodel])
                plot_single_line(df[submetric].mean(), label, color, marker, title, ax, linestyle=submodels[submodel])
                # ax.plot(df.index, df[submetric], color=color, alpha=0.8, linestyle=submodels[submodel])
                # ax.axhline(y=df[metric].mean(), label=f"Avg {label}", color=color, linestyle='--', alpha=0.4)
            else:
                ax.plot(df.index, df[submetric], color=color, alpha=0.15, linestyle=submodels[submodel])
                ax.plot(df.index, smoothed_df[submetric], label=label, color=color, alpha=0.6, linestyle=submodels[submodel])
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        for clientnr in range(len(data)):
            ax.axvline(x=clientnr * epochs_per_client, color='b', linestyle='--', alpha=0.5)
            if show_phases:
                ax.axvline(x=clientnr * epochs_per_client + 15, color='orange', linestyle='--', alpha=0.3)
                ax.axvline(x=clientnr * epochs_per_client + 30, color='orange', linestyle='--', alpha=0.3)

    def plot_multisplit(df, smoothed_df, metric, title, label, ax):
        num_splits = 2
        while f'{metric}/{num_splits}' in df.columns:
            num_splits += 1
        num_splits -= 1
        
        for i in range(1, num_splits + 1):
            submetric = f'{metric}/{i}'
            # for i, submetric in enumerate([f'{metric}/{submodel}' for submodel in list(submodels.keys())]):
            # submodel = list(submodels.keys())[i]
            label += f" Split {i}"
            if 'test' in submetric:
                ax.plot(df.index, df[submetric], color=color, alpha=0.15, linestyle=line_styles[2])
                plot_single_line(df[submetric].mean(), label, color, None, title, ax, linestyle=line_styles[2], alpha=0.25)
            else:
                ax.plot(df.index, df[submetric], color=color, alpha=0.15, linestyle=line_styles[2])
                ax.plot(df.index, smoothed_df[submetric], label=label, color=color, alpha=0.25, linestyle=line_styles[2])
        
        submetric = f'{metric}/split_avg'
        label += f" Split Merged"
        if 'test' in submetric:
            ax.plot(df.index, df[submetric], color=color, alpha=0.15, linestyle=line_styles[0])
            plot_single_line(df[submetric].mean(), label, color, marker, title, ax, linestyle=line_styles[0])
        else:
            ax.plot(df.index, df[submetric], color=color, alpha=0.15, linestyle=line_styles[0])
            ax.plot(df.index, smoothed_df[submetric], label=label, color=color, alpha=0.6, linestyle=line_styles[0])
        
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
    
    # input(df.columns)
    # input(smoothed_df.columns)
    
    for i, (metric, title) in enumerate(metrics.items()):
        
        ax = axs[i // ax_width, i % ax_width] if axs.ndim > 1 else axs[i]
        if metric is None:
            ax.axis('off')
            continue
        if metric + '/MM' in df.columns:
            plot_multisubmodel(df, smoothed_df, metric, title, label, ax)
        
        elif metric +'/1' in df.columns: # Multisplit result (XGBoost)
            plot_multisplit(df, smoothed_df, metric, title, label, ax)
        
        elif metric in df.columns:
            
            # Vertical lines for each client of length epochs
            # if next((i for i, h in enumerate(legend_handles) if h.get_label() == label), None) == 0:
            for clientnr in range(len(data)):
                ax.axvline(x=clientnr * epochs_per_client, color='b', linestyle='--', alpha=0.5)
                    
            
            # If metric is a test metric also plot the average of all clients as a horizontal line
            if 'test' in metric:
                ax.plot(df.index, df[metric], color=color, alpha=0.3)
                plot_single_line(df[metric].mean(), label, color, marker, title, ax)
                # ax.plot(df.index, df[metric], color=color, alpha=0.8)
                # ax.axhline(y=df[metric].mean(), label=f"Avg {label}", color=color, linestyle='--', alpha=0.4)
            else:
                ax.plot(df.index, df[metric], color=color, alpha=0.15)
                ax.plot(df.index, smoothed_df[metric], label=label, color=color, alpha=0.6)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            if label and not "Legend" in metrics: ax.legend()
        elif metric in test:
            plot_single_line(test[metric], label, color, marker, title, ax)
        elif avg_datas and metric + '/MM'  in avg_datas:
            for metric_submodel in [f'{metric}/{submodel}' for submodel in list(submodels.keys())]:
                if metric_submodel in avg_datas:
                    plot_single_line(avg_datas[metric_submodel], label, color, marker, title, ax, linestyle=submodels[metric_submodel.split('/')[-1]])
        elif avg_datas and metric in avg_datas:
           plot_single_line(avg_datas[metric], label, color, marker, title, ax)
        elif metric == 'Legend':
            # Show legend in the last row, first column
            ax.axis('off')
            if legend_handles:
                ax.legend(handles=legend_handles, loc='center', fontsize=12)
        
        elif label is None:
            ax.axis('off')
            ax.set_title(title)
            ax.text(0.5, 0.5, f"{title} not available", fontsize=12, ha='center', va='center')
    
    # ax = axs[2, 3]
    # ax.axis('off')
    # In the last row, last column, show the config data
    if show_config and config_data:  # Only add config on first call    
        ax.set_title("Config Data")
        ax.text(0.5, 0.5, config_data, fontsize=12, ha='center', va='center')
    
    # return axs
    # for i, (metric, title) in enumerate(debug_metrics.items()):
        
    #     ax = axs[i // ax_width, i % ax_width]
    #     if metric is None:
    #         ax.axis('off')
    #         continue
    #     if metric + '/CLAM' in df.columns:
    #         for i, submetric in enumerate([f'{metric}/{submodel}' for submodel in ["CLAM", "CD", "Fusion", "MM"]]):
    #             submodel = list(submodels.keys())[min(i,2)]
    #             label += f" {submetric}"
    #             if submetric in df.columns:
    #                 value = df[submetric]
    #                 # print("METRIC:", metric, submetric, value)
    #                 if metric == 'Lossweights':
    #                     print("USES EXPONENTIAL")
    #                     value = np.exp(value)
    #                 ax.plot(df.index, value, alpha=0.8, linestyle=submodels[submodel], label=submetric)
            
    #         ax.set_title(title)
    #         ax.set_xlabel('Epoch')
    #         ax.set_ylabel(title)
    #         ax.legend()
    
    # ax = axs[2, 3]
    # ax.axis('off')
    # In the last row, last column, show the config data
    if show_config and config_data:  # Only add config on first call    
        ax.set_title("Config Data")
        ax.text(0.5, 0.5, config_data, fontsize=12, ha='center', va='center')
    
    return axs

def merge_splits(datas: dict):
    # print(datas.keys())
    # print(list(datas.values())[0].keys())
    # input(list(list(datas.values())[0].values())[0])
    
    merged_metrics = {}
    
    for experiment_name, data in datas.items():
        for clientnr, client_data in data.items():
            for metric in client_data.keys():
                if not metric.endswith('/1'):
                    continue
                # Merge all clients for this metric into a single list
                merged_metric = metric[:-2]+ '/split_avg'
                
                
                values = []
                
                for metric_compare in client_data.keys():
                    if metric[:-2] == metric_compare[:-2]:
                        values.extend(client_data[metric_compare])
            
                # Replace the original metric with the merged one
                merged_metrics[merged_metric] = values
            for merged_metric, values in merged_metrics.items():
                client_data[merged_metric] = [np.mean(values)]
    return datas
                


def smooth_data(data, window_size=5):
    """
    Apply a moving average smoothing to the data dictionary.
    
    Args:
        data: Dictionary with format {client: {metric: [value1, value2, ...]}}
        window_size: Size of the moving average window
        
    Returns:
        Smoothed data dictionary with the same structure
    """
    import numpy as np
    import pandas as pd
    
    smoothed_data = {}
    
    for client, client_data in data.items():
        smoothed_data[client] = {}
        for metric, values in client_data.items():
            # Convert to pandas Series for smoothing
            series = pd.Series(values)
            # Apply moving average
            smoothed_series = series.rolling(window=window_size, min_periods=1).mean()
            smoothed_data[client][metric] = smoothed_series.tolist()
    return smoothed_data

def datadict_to_df(data: dict):
    # Dict has format {client: {metric: [value1, value2, ...]}}
    df_data = {}
    
    debug_list = {client_nr: list(client_data.keys()) for client_nr, client_data in data.items()}        
    # input(f"Debug: {debug_list}")
    
    for clientnr, client_data in data.items():
        
        for metric, values in client_data.items():
            if metric not in df_data:
                df_data[metric] = []
            df_data[metric].extend(values)
    
    return pd.DataFrame(df_data)
    
def tensorboard_to_datadict(experiment_name: str, exp_dir:str = ROOT_RUNS, experiment:str = None):
    """Extract data from TensorBoard logs for custom plotting"""
    
    if experiment is not None:
        log_dir = experiment
    else:
        experiment_dir = os.path.join(exp_dir, experiment_name)
        log_dir = os.path.join(experiment_dir, "log")
        log_dir_results = os.path.join(ROOT_RESULTS, experiment_name, "log")
        log_dir_delete = os.path.join(ROOT_DELETE, experiment_name, "log")
        
        assert os.path.exists(log_dir) or os.path.exists(log_dir_results) or os.path.exists(log_dir_delete), f"Log directory {log_dir} nor {log_dir_results} exist."
        if not os.path.exists(log_dir):
            log_dir = log_dir_results
        if not os.path.exists(log_dir):
            log_dir = log_dir_delete
        
    # Dict to store data: {client: {metric: [values_for_all_epochs]}}
    data = defaultdict(lambda: defaultdict(list))
    test = {}
    
    # Track the maximum step seen for each client and metric
    max_steps = defaultdict(lambda: defaultdict(int))
    
    
    
    for root, dirs, files in os.walk(log_dir):
        for clientnr, client in enumerate(sorted(dirs)):
            if not client.startswith("client_") and not client == "test":
                print(f"Skipping {client} as it is not a client")
                continue
            client_path = os.path.join(root, client)
            for file in os.listdir(client_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(client_path, file)
                    for event in tf.compat.v1.train.summary_iterator(event_file):
                        if client == "test":
                            for value in event.summary.value:
                                test[value.tag] = value.simple_value
                            continue
                        
                        for value in event.summary.value:
                            # Track the maximum step for this metric
                            max_steps[clientnr][value.tag] = max(max_steps[clientnr][value.tag], event.step + 1)
                            
                            # Ensure the list is long enough
                            while len(data[clientnr][value.tag]) <= event.step:
                                data[clientnr][value.tag].append(None)
                            
                            # Add the value at the correct position
                            data[clientnr][value.tag][event.step] = value.simple_value
    
    # Ensure all metric lists have consistent length within each client
    for clientnr in data:
        max_length = max(max_steps[clientnr].values(), default=0)
        for metric in data[clientnr]:
            assert len(data[clientnr][metric]) <= max_length, f"Data length mismatch for client {clientnr}, metric {metric}"
    
    return data, test


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
                            
        elif item.startswith("client_server"):
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


def plot_federated_experiment(experiment_name: str, submodel: str = 'MM'):
    """
    Plot federated learning experiment results
    
    Args:
        experiment_name: Name of the federated experiment
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD', or 'ALL')
    """
    # Get federated data
    client_data, server_data = tensorboard_to_datadict_federated(experiment_name)
    
    # Create plot
    fig, axs = create_federated_plot()
    fig.suptitle(f"Federated Learning: {experiment_name}\nClient Training vs Global Model Performance ({submodel})", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Colors for different clients
    client_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    ax_dims = axs.shape
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    
    for i, (base_metric, title) in enumerate(federated_metrics.items()):
        if base_metric is None or title is None:
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
                                    client_colors, submodels.get(sub, '-'))
        else:
            metric = f"{base_metric}/{submodel}"
            plot_federated_metric(ax, metric, title, client_data, server_data, client_colors)
    
    # Hide empty subplots
    total_plots = len([m for m in federated_metrics.keys() if m is not None])
    if hasattr(axs, 'flat'):
        for j in range(total_plots, len(axs.flat)):
            axs.flat[j].axis('off')
    
    # Add legend for the last subplot
    if hasattr(axs, 'flat') and len(axs.flat) > total_plots:
        legend_ax = axs.flat[-1]
        legend_ax.axis('off')
        
        # Create legend handles
        legend_handles = []
        for client_id in sorted(client_data.keys()):
            color = client_colors[client_id % len(client_colors)]
            line = plt.Line2D([0], [0], color=color, linewidth=2, label=f'Client {client_id} Training')
            legend_handles.append(line)
        
        # Add global model legend
        server_line = plt.Line2D([0], [0], color='red', marker='s', markersize=8, 
                               linestyle=':', linewidth=2, label='Global Model Performance')
        legend_handles.append(server_line)
        
        # Add round boundary legend
        boundary_line = plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, 
                                 label='Round Boundaries')
        legend_handles.append(boundary_line)
        
        legend_ax.legend(handles=legend_handles, loc='center', fontsize=12)
        legend_ax.set_title('Legend')
    
    plt.tight_layout()
    
    # Save the figure
    plot_dir = os.path.join(ROOT_RESULTS, experiment_name)
    plot_path = os.path.join(plot_dir, f"federated_plot_{submodel.lower()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


def plot_federated_metric(ax, metric, title, client_data, server_data, client_colors, linestyle='-'):
    """Plot a single metric for federated learning"""
    
    # Get all rounds that exist
    all_rounds = set()
    for client_id in client_data.keys():
        all_rounds.update(client_data[client_id].keys())
    all_rounds = sorted(all_rounds)
    
    max_steps_per_round = 0
    # Find maximum steps per round across all clients
    for client_id in client_data.keys():
        for round_num in all_rounds:
            if round_num in client_data[client_id] and metric in client_data[client_id][round_num]:
                steps_in_round = len([v for v in client_data[client_id][round_num][metric] if v is not None])
                max_steps_per_round = max(max_steps_per_round, steps_in_round)
    
    # Plot each client
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
    
    # Add vertical lines for round boundaries
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
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            ax2.scatter(server_rounds, server_values, color='red', marker='s', 
                      s=100, alpha=0.9, label='Global Model (Server)', zorder=5, edgecolor='black')
            ax2.plot(server_rounds, server_values, color='red', alpha=0.6, 
                   linestyle=':', linewidth=2, zorder=4)
            ax2.set_ylabel(title + ' (Global Model)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Set x-axis to show round numbers
            ax2.set_xlim(-0.5, max(server_rounds) + 0.5)
            ax2.set_xticks(server_rounds)
            ax2.set_xticklabels([f'Round {r}' for r in server_rounds])
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
    
    # Add round labels on x-axis
    if all_rounds and max_steps_per_round > 0:
        round_centers = [(i + 0.5) * max_steps_per_round for i in range(len(all_rounds))]
        ax.set_xticks(round_centers)
        ax.set_xticklabels([f'Round {r}' for r in all_rounds])
    
    ax.set_title(title)
    ax.set_xlabel('Federated Learning Rounds')
    ax.set_ylabel(title)
    
    # Improve legend handling - only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, alpha=0.3)


def plot_federated_comparison(experiment_names: list, submodel: str = 'MM', metric_filter: str = 'test'):
    """
    Compare multiple federated learning experiments
    
    Args:
        experiment_names: List of federated experiment names to compare
        submodel: Which submodel to focus on ('MM', 'CLAM', 'CD')
        metric_filter: Which type of metrics to plot ('train', 'val', 'test', or 'all')
    """
    
    # Filter metrics based on metric_filter
    if metric_filter == 'all':
        metrics_to_plot = {k: v for k, v in federated_metrics.items() if k is not None and v is not None}
    else:
        metrics_to_plot = {k: v for k, v in federated_metrics.items() 
                          if k is not None and v is not None and metric_filter in k}
    
    # Create plot
    fig, axs = create_federated_plot(len(metrics_to_plot))
    fig.suptitle(f"Federated Learning Comparison ({submodel} - {metric_filter} metrics)", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Colors for different experiments
    exp_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Get data for all experiments
    all_data = {}
    for exp_name in experiment_names:
        try:
            client_data, server_data = tensorboard_to_datadict_federated(exp_name)
            all_data[exp_name] = (client_data, server_data)
        except Exception as e:
            print(f"Warning: Could not load data for {exp_name}: {e}")
            continue
    
    ax_dims = axs.shape if hasattr(axs, 'shape') else (1, 1)
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    
    for i, (base_metric, title) in enumerate(metrics_to_plot.items()):
        # Get the subplot
        if len(ax_dims) > 1:
            ax = axs[i // ax_width, i % ax_width]
        else:
            ax = axs[i] if hasattr(axs, '__len__') else axs
        
        metric = f"{base_metric}/{submodel}"
        
        # Plot each experiment
        for exp_idx, (exp_name, (client_data, server_data)) in enumerate(all_data.items()):
            color = exp_colors[exp_idx % len(exp_colors)]
            
            # Plot server evaluation points only for comparison clarity
            server_values = []
            server_rounds = []
            for round_num in sorted(server_data.keys()):
                if metric in server_data[round_num]:
                    server_values.append(server_data[round_num][metric])
                    server_rounds.append(round_num)
            
            if server_values:
                ax.plot(server_rounds, server_values, color=color, marker='o', 
                       linewidth=2, markersize=6, label=exp_name, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel('Federated Round')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        
        # Add legend if there are multiple experiments
        if len(all_data) > 1:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
    
    # Hide empty subplots
    total_plots = len(metrics_to_plot)
    if hasattr(axs, 'flat'):
        for j in range(total_plots, len(axs.flat)):
            axs.flat[j].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(ROOT_RESULTS, f"federated_comparison_{submodel.lower()}_{metric_filter}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    plt.show()


def get_config_data(experiment_name: str):
    # Config data is stored in "config.txt" in the experiment directory
    experiment_dir = os.path.join(ROOT_RUNS, experiment_name)
    config_path = os.path.join(experiment_dir, "config.txt")
    assert os.path.exists(config_path), f"Config file {config_path} does not exist."
    # Return as string
    config_data = ""
    with open(config_path, 'r') as f:
        for line in f:
            config_data += line.strip() + "\n"
    return config_data

def create_plot():
    # Create the figure and axes
    num_metrics = len(metrics)
    '''
    Num metrics -> rows, columns
    1 -> 1,1
    2 -> 1,2
    3 -> 1,3
    4 -> 2,2
    5 -> 2,3
    6 -> 2,3
    7 -> 3,3
    8 -> 3,3
    9 -> 3,3
    10 -> 3,4
    11 -> 3,4
    12 -> 3,4
    13 -> 4,4
    14 -> 4,4
    15 -> 4,4
    16 -> 4,4
    '''
    rows = {
        1:[1],
        2:[2,3,4,5,6],
        3:[7,8,9,10,11,12],
        4:[13,14,15,16]
    }
        
    num_rows = [i for i,n in rows.items() if num_metrics in n][0]
    num_cols = (num_metrics + num_rows - 1) // num_rows  # Ceiling division to get number of columns
    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    return fig, axs

def plot_experiment(experiment_name: str):
    # Get data
    data, test = tensorboard_to_datadict(experiment_name)
    # Get config data
    config_data = get_config_data(experiment_name)
    # Plot
    
    fig, axs = create_plot()
    # fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Experiment: {experiment_name}", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)    
    
    axs = plot_run(data, experiment_name, config_data=config_data, axs=axs, test=test)
    
    # Save the figure
    plt.tight_layout()
    plot_dir = os.path.join(ROOT_RUNS, experiment_name)
    plot_path = os.path.join(plot_dir, "plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()
    

def plot_experiments(experiment_names: list, smooth_window_size=10, max_k = 10):
    datas = {}
    # config_datas = {}
    tests = {}
    
    fig, axs = create_plot()
    fig.suptitle(f"Experiment Comparison", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4) 
    
    for experiment_name in experiment_names:
        if experiment_name.endswith('_splits'):
            for experiment in os.listdir(os.path.join(ROOT_RESULTS, experiment_name)):
                data, test = tensorboard_to_datadict(experiment = experiment)
                datas[experiment] = data
                tests[experiment] = test
            continue
        data, test = tensorboard_to_datadict(experiment_name)
        # config_data = get_config_data(experiment_name)
        # Extract model name from config data
        
        datas[experiment_name] = data
        # config_datas[experiment_name] = config_data
        tests[experiment_name] = test
    
    datas = merge_splits(datas)
    # # datas dictionary format is {experiment_name: {client: {metric: [value1, value2, ...]}}}
    # # Configure data to have same number of epochs per client
    # nr_clients = len(datas[list(datas.keys())[0]])
    # # Assert that all experiments have the same number of clients
    # for experiment_name, data in datas.items():
    #     assert len(data) == nr_clients, f"Experiment {experiment_name} has {len(data)} clients, expected {nr_clients} clients."
    
    max_nr_clients = max(len(data) for data in datas.values())
    max_nr_clients = min(max_nr_clients, max_k)  # Limit to max_k clients
    # Delete clients that are greater than max_k
    for experiment_name, data in datas.items():
        for clientnr in list(data.keys()):
            if clientnr >= max_k:
                del data[clientnr]
    
    # Make sure all clients have the same number of epochs, also adjust that all clients have the same length
    max_length = 0
    for experiment_name, data in datas.items():
        for clientnr, client_data in data.items():
            if clientnr >= max_k:
                continue
            # Find the maximum length of epochs in this client
            local_max_length = max(max_length, max(len(values) for values in client_data.values()))
            max_length = max(max_length, local_max_length)
            
    for clientnr in range(max_nr_clients):
        for experiment_name, data in datas.items():
            # Ensure each client has the same number of epochs by padding with last value
            if clientnr not in data:
                continue  # Skip if client does not exist in this experiment
            for metric in data[clientnr]:
                while len(data[clientnr][metric]) < max_length:
                    data[clientnr][metric].append(data[clientnr][metric][-1] if data[clientnr][metric] else None)
    
    avg_datas = None
    # If any metric starts with 'Avg_', calculate the average of the last epochs
    if any(metric.startswith('Avg_') for metric, _ in metrics.items() if metric):
        avg_datas = calc_avg_val_metrics(datas)
    
    
    
    smoothed_datas = {}       
    # Smooth the data
    for experiment_name, data in datas.items():
        smoothed_datas[experiment_name] = smooth_data(data, window_size=smooth_window_size)
    
    
    
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', '*', 'x', '+', 'v', '>', '<', 'p', 'h']
    
    # Create artificial legend
    import matplotlib.lines as mlines
    legend_handles = []
    for i, (model_name, data) in enumerate(datas.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='-', label=model_name)
        legend_handles.append(handle)
    
    if len(submodels) > 1:
        for submodel, linestyle in submodels.items():
            # Add submodel legend handles
            handle = mlines.Line2D([], [], color='black', linestyle=linestyle, label=f"{submodel} (submodel)")
            legend_handles.append(handle)
    
    
    for i, (model_name, data) in enumerate(datas.items()):
        # config_data = config_datas[model_name]
        test = tests[model_name]
        axs = plot_run(data, smoothed_datas[model_name], model_name, config_data=None, axs=axs, label=model_name, show_config=False, test=test, color = colors[i], marker = markers[i], avg_datas=avg_datas[model_name] if avg_datas else None, legend_handles=legend_handles)
    # Plot all models in one plot
    # Save the figure
    plt.tight_layout()
    # plot_dir = os.path.join(ROOT_RUNS, experiment_name)
    # plot_path = os.path.join(plot_dir, "plot.png")
    # plt.savefig(plot_path, dpi=300)
    # print(f"Plot saved to {plot_path}")
    plt.show()

def calc_avg_val_metrics(data: dict):
    """
    Calculate average validation metrics of last epochs from all clients.
    
    Args:
        data: Dictionary with format {client: {metric: [value1, value2, ...]}}
        
    Returns:
        Dictionary with average validation metrics.
    """
    avg_metrics = {}
    # Metrics to average are those that start with 'Avg_'
    avg_metrics_keys = [metric for metric, _ in metrics.items() if metric and metric.startswith('Avg_')]
    # Add /MM, /CLAM, /CD to the metrics
    avg_metrics_keys += [f'{metric}/{submodel}' for metric, _ in metrics.items() if metric and metric.startswith('Avg_') for submodel in list(submodels.keys())]
    
    
    for experiments_name, exp_data in data.items():
        avg_metrics[experiments_name] = {}
        for clientnr, client_data in exp_data.items():
            for metric in avg_metrics_keys:
                if metric not in avg_metrics[experiments_name]:
                    avg_metrics[experiments_name][metric] = []
                # Get the last value of the metric for this client
                not_avg_metric = metric.replace('Avg_', '')
                if not_avg_metric in client_data:
                    last_value = client_data[not_avg_metric][-1] if client_data[not_avg_metric] else None
                    avg_metrics[experiments_name][metric].append(last_value)
        # Calculate the average of the last values for each metric
        for metric in avg_metrics[experiments_name]:
            if avg_metrics[experiments_name][metric]:
                avg_metrics[experiments_name][metric] = sum(avg_metrics[experiments_name][metric]) / len(avg_metrics[experiments_name][metric])
            else:
                avg_metrics[experiments_name][metric] = None
        # Delete metrics with None values
        avg_metrics[experiments_name] = {k: v for k, v in avg_metrics[experiments_name].items() if v is not None}
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting script for TensorBoard logs')
    parser.add_argument('--name', type=str, help='Name of the experiment to plot')
    parser.add_argument('--names', nargs='+', help='Names of experiments to compare (for federated comparison mode)')
    parser.add_argument('--max_k', type=int, default=10, help='Maximum number of clients to plot')
    parser.add_argument('--mm_only', action='store_true', help='Plot only MM experiments')
    parser.add_argument('--clam_only', action='store_true', help='Plot only CLAM experiments')
    parser.add_argument('--federated', action='store_true', help='Use federated learning plotting mode')
    parser.add_argument('--compare', action='store_true', help='Compare multiple federated experiments')
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD', 'ALL'], 
                       help='Which submodel to plot for federated learning')
    parser.add_argument('--metric_filter', type=str, default='test', choices=['train', 'val', 'test', 'all'],
                       help='Which metrics to show in comparison mode')
    args = parser.parse_args()

    experiment_name = args.name if args.name else 'test2_2_client_s1'
    max_k = args.max_k if args.max_k else 10
    
    if args.mm_only: submodels = {'MM': '-'}
    elif args.clam_only: submodels = {'CLAM': '-'}
    
    # Use federated comparison if requested
    if args.compare and args.names:
        plot_federated_comparison(args.names, args.submodel, args.metric_filter)
    # Use federated plotting if requested
    elif args.federated or args.name == 'test2_2_client_s1':  # Auto-detect federated for test experiments
        plot_federated_experiment(experiment_name, args.submodel)
    else:
        # Original plotting functions for backward compatibility
        
        # plot_experiment(experiment_name)
        # plot_experiments(['CDSoloNewCLAMSplit100', 'CDSoloNewCLAMSplit200', 'CDSoloNewCLAMSplit250', 'CDSoloNew03test'], smooth_window_size=20)
        # plot_experiments(['CDSoloSmallOld', 'CDLogReg', 'CDXGBoost', 'CDRanFor', 'CDSoloNew10', 'CDSolo10_0_1', 'CDSolotest'])
        # plot_experiments(['CDLogReg', 'CDXGBoost', 'CDRanFor','CDSolo10_0_1', 'CDSolotest', 'CDSolotestNoEarly', 'CDSolotestNoEarly0403', 'CDSolotest0403'])
        # plot_experiments(['CDSolo10_0_1','CDSolotest', 'CDSolotest0403', 'CDSoloLogRegNew'])
        # plot_experiments(['CDSolotest0403','CDSolotestNoEarly0403', 'CDSolo10_0_1', 'CDSolo0404_1'])
        # plot_experiments(['CDSoloNew', 'CDSoloSmallOld', 'CDSoloSmallNew', 'CDSoloNew2', 'CDSoloNew3'])
        
        
        # plot_experiments(['WholeImgResNet', 'SimplerCNN', 'SimpleMIL3', 'SimpleMIL4', 'CLAM_Img_only', 'chimera_CLAM_new3_s1'])
        # plot_experiments(['chimera_MM_s1', 'chimera_MM_Pretrained_s1', 'grid_search_Winner', 'chimera_New_MM_s1', 'chimera_New_Img_Only_s1'])#,'CDSoloNew'])
        # plot_experiments(['chimera_MM_Renewed_s1', 'chimera_MM_Pretrained_Renewed_s1', 'chimera_CLAM_Img_Renewed_s1', 'chimera_MM_Freezed_Renewed_s1', 'CDSoloNew'])
        # plot_experiments(['chimera_New_MM_s1', 'chimera_New_Img_Only2Full_s1', 'chimera_New_Img_Only_s1', 'CLAM_Img_only', 'chimera_New_Img_Tiny_No_Gated_s1'])#,'CDSoloNew'])
        # plot_experiments(['CDSoloNew', 'CDRanFor', 'CDXGBoost','CDLogReg'])
        
        # plot_experiments(['chimera_New_Img_Tiny_Gated_MM_1F_pretrained3_s1'])
        
        # Example federated experiment - test with default name
        print("Testing federated plotting with default experiment...")
        plot_federated_experiment(experiment_name, 'MM')
        
        # for i in range(1,6):
        #     plot_experiments([
        #         f'MMWM544364_uni_v2_page_1_split_{i}_s1',
        #         f'grid_search_MM_1_004_split_{i}_s1',
        #         f'grid_search_MM_30-10-34_0_005_split_{i}_s1',
        #         f'grid_search_MM_01-17-32_0_002_split_{i}_s1',
        #     ])