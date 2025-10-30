import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
from collections import defaultdict
import argparse
import numpy as np
import ast

ROOT_RUNS = "/gris/gris-f/homelv/phempel/masterthesis/MMFL/runs"
ROOT_RESULTS = "/gris/gris-f/homelv/phempel/masterthesis/MMFL/results"
ROOT_DELETE = "/gris/gris-f/homelv/phempel/masterthesis/MMFL/results_delete"
ROOT_FAILED = "/gris/gris-f/homelv/phempel/masterthesis/MMFL/results_delete"

metrics = {
    
    # 'Accuracy/train': 'Training Accuracy',
    # 'Binary_Accuracy/train': 'Training Binary Accuracy',
    # # 'ROC_AUC/train': 'Training ROC AUC',
    # 'Hyp_diffs': None,
    # # None: None,
    # 'F1/train': 'Training F1 Score',
    # # 'Loss/train': 'Training Loss',
    # # 'Accuracy/val': 'Validation Accuracy',
    # # None: None,
    # 'Binary_Accuracy/val': 'Validation Binary Accuracy',
    
    'Legend': None,
    # 'Hyp_diffs': None,
    # # 'ROC_AUC/val': 'Validation ROC AUC',
    # 'F1/val': 'Validation F1 Score',
    
    
    # # 'Loss/val': 'Validation Loss',
    # # 'Avg_Binary_Accuracy/val': 'Average Validation Binary Accuracy',
    # # 'Avg_ROC_AUC/val': 'Average Validation ROC AUC',
    # 'Avg_F1/val': 'Average Validation F1 Score',
    # # None: None,
    # # 'Accuracy/test': 'Test Accuracy',
    'Binary_Accuracy/test': 'Test Binary Accuracy',
    'ROC_AUC/test': 'Test ROC AUC',
    'F1/test': 'Test F1 Score',
    # None: None,
}

line_styles = ['-', '--', '-.', ':']
submodels = {
    'CLAM': line_styles[1],
    'CD': line_styles[2],
    'MM': line_styles[0]
}



def plot_run(data: dict, axs=None, label=None, legend_handles=None, colors=None, markers = None, hyp_diffs=None): 
    ax_dims = axs.shape
    ax_width = ax_dims[1] if len(ax_dims) > 1 else 1
    
    print(data.keys(), list(data.values())[0].keys())
    
    for i, (metric, title) in enumerate(metrics.items()):
        ax = axs[i // ax_width, i % ax_width] if axs.ndim > 1 else axs[i]
        if title is None:
            ax.axis('off')
        if metric is None:
            # Skip None metrics
            continue
        elif metric == 'Legend':
            # Show legend in the last row, first column
            if legend_handles:
                ax.legend(handles=legend_handles, loc='center', fontsize=12)
        elif metric == 'Hyp_diffs':
            text = "\n".join(f"{i}: {diff}" for i, diff in enumerate(hyp_diffs)) if hyp_diffs else "No hyperparameter differences"
            ax.text(0.5, 0.5, text, fontsize=9, ha='center', va='center', transform=ax.transAxes)                
        else:
            for j, submetric in enumerate([f'{metric}/{submodel}' for submodel in list(submodels.keys())]):
                # Plot bar plot for the given metric
                x = [f"{k}/{list(submodels.keys())[j]}" for k, exp in enumerate(data.keys())]
                # x = [list(submodels.keys())[i] for exp in data.keys()]
                y = [data[exp][submetric][0] if submetric in data[exp] else 0 for exp in data.keys()]
                vals = [data[exp][submetric][2:] if submetric in data[exp] else [] for exp in data.keys()]
                if no_std:
                    e = [0 for exp in data.keys()] 
                else:
                    e = [data[exp][submetric][1] if submetric in data[exp] else 0 for exp in data.keys()]
                
                # Create bar plot with error bars
                for i in range(len(data.keys())):
                    ax.errorbar(x[i], y[i], yerr=e[i], alpha=0.7, linestyle='None', marker=markers[i%len(markers)], color=colors[i%len(colors)], label=submetric)
                    xs = [x[i]] * len(vals[i])  # x-coordinates for scatter points
                    ax.scatter(xs, vals[i], label=submetric, s=20, alpha=0.3, color = colors[i % len(colors)])
                    
                    # Add text annotation with F1 score and standard deviation
                    text_y = y[i] + e[i] + 0.01  # Position text slightly above the error bar
                    ax.text(x[i], text_y, f'{y[i]:.3f}±{e[i]:.3f}', 
                           ha='center', va='bottom', fontsize=8, rotation=45)
                
                
                    
                    # Plotting the bar with error bars
                # ax.errorbar(x, y, yerr=e, alpha=0.7, linestyle='None', marker = )#, edgecolor='black', capsize=5)#, color = colors[:len(data.keys())])
                # ax.bar(x, y, yerr=e, alpha=0.7, label=submetric, color=colors[j % len(colors)], edgecolor='black', capsize=5)
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Experiments', fontsize=12)
                ax.set_ylabel('Value', fontsize=12)
            
    
    # Create bar plot for each metric


def merge_splits_avg_std(list_of_datas: list, max_k = 10):
    merged_data = defaultdict(list)
    merged_data_fold_avg = defaultdict(list)
    
    for data in list_of_datas:
        merged_data_of_split = defaultdict(list)
        for i, fold_data in enumerate(data.values()):
            if i >= max_k:
                continue
            for metric, values in fold_data.items():
                # merged_data[metric].append(values[-1])
                merged_data_of_split[metric].append(values[-1])  # Take the last value of each metric for this fold
        
        for metric, values in merged_data_of_split.items():
            merged_data[metric].extend(values)  # Merge all folds for this metric into a single list
            merged_data_fold_avg[metric].append(np.mean(values))  # Store the average of the last values for this metric across folds
        
    
    merged_avg_std = {}
    
    for metric, values in merged_data.items():
        if values:  # Check if there are values to compute avg/std
            avg_value = np.mean(values)
            std_value = np.std(values)
            merged_avg_std[metric] = [avg_value, std_value]+ merged_data_fold_avg[metric]
    
    return merged_avg_std

def merge_splits(datas: dict):
    # print(datas.keys())
    # print(list(datas.values())[0].keys())
    # input(list(list(datas.values())[0].values())[0])
    
    merged_metrics = {}
    
    for experiment_name, data in datas.items():
        for foldnr, fold_data in data.items():
            for metric in fold_data.keys():
                if not metric.endswith('/1'):
                    continue
                # Merge all folds for this metric into a single list
                merged_metric = metric[:-2]+ '/split_avg'
                
                
                values = []
                
                for metric_compare in fold_data.keys():
                    if metric[:-2] == metric_compare[:-2]:
                        values.extend(fold_data[metric_compare])
            
                # Replace the original metric with the merged one
                merged_metrics[merged_metric] = values
            for merged_metric, values in merged_metrics.items():
                fold_data[merged_metric] = [np.mean(values)]
    return datas
                


def smooth_data(data, window_size=5):
    """
    Apply a moving average smoothing to the data dictionary.
    
    Args:
        data: Dictionary with format {fold: {metric: [value1, value2, ...]}}
        window_size: Size of the moving average window
        
    Returns:
        Smoothed data dictionary with the same structure
    """
    import numpy as np
    import pandas as pd
    
    smoothed_data = {}
    
    for fold, fold_data in data.items():
        smoothed_data[fold] = {}
        for metric, values in fold_data.items():
            # Convert to pandas Series for smoothing
            series = pd.Series(values)
            # Apply moving average
            smoothed_series = series.rolling(window=window_size, min_periods=1).mean()
            smoothed_data[fold][metric] = smoothed_series.tolist()
    return smoothed_data

def datadict_to_df(data: dict):
    # Dict has format {fold: {metric: [value1, value2, ...]}}
    df_data = {}
    
    debug_list = {fold_nr: list(fold_data.keys()) for fold_nr, fold_data in data.items()}        
    # input(f"Debug: {debug_list}")
    
    for foldnr, fold_data in data.items():
        
        for metric, values in fold_data.items():
            if metric not in df_data:
                df_data[metric] = []
            df_data[metric].extend(values)
    
    return pd.DataFrame(df_data)
    
def tensorboard_to_datadict(experiment_name: str = None, exp_dir:str = ROOT_RUNS, experiment:str = None):
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
        
    # Dict to store data: {fold: {metric: [values_for_all_epochs]}}
    data = defaultdict(lambda: defaultdict(list))
    
    # Track the maximum step seen for each fold and metric
    max_steps = defaultdict(lambda: defaultdict(int))
    
    
    for root, dirs, files in os.walk(log_dir):
        for foldnr, fold in enumerate(sorted(dirs)):
            if not fold.startswith("fold_") and not fold == "test":
                print(f"Skipping {fold} as it is not a fold")
                continue
            fold_path = os.path.join(root, fold)
            for file in os.listdir(fold_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(fold_path, file)
                    for event in tf.compat.v1.train.summary_iterator(event_file):
                        
                        for value in event.summary.value:
                            # Track the maximum step for this metric
                            max_steps[foldnr][value.tag] = max(max_steps[foldnr][value.tag], event.step + 1)
                            
                            # Ensure the list is long enough
                            while len(data[foldnr][value.tag]) <= event.step:
                                data[foldnr][value.tag].append(None)
                            
                            # Add the value at the correct position
                            data[foldnr][value.tag][event.step] = value.simple_value
    
    # Ensure all metric lists have consistent length within each fold
    for foldnr in data:
        max_length = max(max_steps[foldnr].values(), default=0)
        for metric in data[foldnr]:
            assert len(data[foldnr][metric]) <= max_length, f"Data length mismatch for fold {foldnr}, metric {metric}"
    
    return data


def get_config_data(experiment_dir, experiment_name):
    config_path = os.path.join(experiment_dir, experiment_name, f"experiment_{experiment_name[:-3]}.txt")
    assert os.path.exists(config_path), f"Config file {config_path} does not exist."
    # Return as string
    config_data = ""
    with open(config_path, 'r') as f:
        config_data = f.read()
    # Convert to dict
    return ast.literal_eval(config_data)

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


def plot_experiments(experiment_names: list, smooth_window_size=10, max_k = 10):
    datas = {}
    # config_datas = {}
    tests = {}
    
    fig, axs = create_plot()
    fig.suptitle(f"Experiment Comparison", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4) 
    config_datas = []
    
    for experiment_name in experiment_names:
        datas_list = []
        for i, experiment in enumerate(os.listdir(os.path.join(ROOT_RESULTS, experiment_name))):
            # if i == 2: 
            #     print("SKIPPING FOLD 3!!!")
            #     continue
            
            data = tensorboard_to_datadict(experiment_name=experiment, exp_dir=os.path.join(ROOT_RESULTS, experiment_name))
            datas_list.append(data)
        config_datas.append(get_config_data(os.path.join(ROOT_RESULTS, experiment_name), experiment))
        # Merge those splits to average and variance
        datas[experiment_name] = merge_splits_avg_std(datas_list, max_k=max_k)
        
        # remove from each config data entry the key value pairs that are equal for all experiments
    relevant_keys = []
    config_data = config_datas[0]
    for key in config_data.keys():
        try:
            if not all(config_data[key] == config_datas[i][key] for i in range(1, len(config_datas))):
                relevant_keys.append(key)
        except KeyError:
            continue
    if 'results_dir' in relevant_keys: relevant_keys.remove('results_dir')
    if 'experiment' in relevant_keys: relevant_keys.remove('experiment')
    if 'split_dir' in relevant_keys: relevant_keys.remove('split_dir')
    if 'task' in relevant_keys: relevant_keys.remove('task')
    # relevant_keys.remove('model_type')
    
    config_datas = [{k: v for k, v in config_data.items() if k in relevant_keys} for config_data in config_datas]
        
      
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']    
    markers = ['o', 's', '^', 'D', '*', 'x', '+', 'v', '>', '<', 'p', 'h']
    
    # Create artificial legend with F1 score and variance
    import matplotlib.lines as mlines
    legend_handles = []
    for i, (model_name, data) in enumerate(datas.items()):
        color = colors[i % len(colors)]
        # Try to get F1/test score and std
        f1_label = ""
        f1_key = 'F1/test/MM' if 'F1/test/MM' in data else None
        if not f1_key:
            # Try to find any F1/test key
            for k in data.keys():
                if k.startswith('F1/test'):
                    f1_key = k
                    break
        if f1_key and isinstance(data[f1_key], list) and len(data[f1_key]) >= 2:
            f1_val = data[f1_key][0]
            f1_std = data[f1_key][1]
            f1_label = f" (F1: {f1_val:.3f}±{f1_std:.3f})"
        handle = mlines.Line2D([], [], color=color, linestyle='-', label=model_name + f1_label)
        legend_handles.append(handle)
    
    plot_run(datas, axs=axs, label=model_name, colors = colors, legend_handles=legend_handles, markers=markers, hyp_diffs=config_datas)
    
    

    # Save the figure
    plt.tight_layout()
    # plot_dir = os.path.join(ROOT_RUNS, experiment_name)
    # plot_path = os.path.join(plot_dir, "plot.png")
    # plt.savefig(plot_path, dpi=300)
    # print(f"Plot saved to {plot_path}")
    plt.show()

def calc_avg_val_metrics(data: dict):
    """
    Calculate average validation metrics of last epochs from all folds.
    
    Args:
        data: Dictionary with format {fold: {metric: [value1, value2, ...]}}
        
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
        for foldnr, fold_data in exp_data.items():
            for metric in avg_metrics_keys:
                if metric not in avg_metrics[experiments_name]:
                    avg_metrics[experiments_name][metric] = []
                # Get the last value of the metric for this fold
                not_avg_metric = metric.replace('Avg_', '')
                if not_avg_metric in fold_data:
                    last_value = fold_data[not_avg_metric][-1] if fold_data[not_avg_metric] else None
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
    parser.add_argument('--max_k', type=int, default=10, help='Maximum number of folds to plot')
    parser.add_argument('--mm_only', action='store_true', help='Plot only MM experiments')
    parser.add_argument('--clam_only', action='store_true', help='Plot only CLAM experiments')
    parser.add_argument('--no_std', action='store_true', help='Do not plot standard deviation')
    args = parser.parse_args()

    experiment_name = args.name if args.name else '2025-05-22_11-31-27'
    max_k = args.max_k if args.max_k else 10
    no_std = args.no_std if args.no_std else False
    
    if args.mm_only: submodels = {'MM': '-'}
    elif args.clam_only: submodels = {'CLAM': '-'}
    
    plot_experiments([
        # 'MMWM544364_uni_v2_page_1_splits', #!
        # 'grid_search_MM_1_004_splits',
        # 'grid_search_MM_orig_splits',
        # 'grid_search_MM_0_000_splits',
        # 'grid_search_MM_0_001_splits',
        # 'grid_search_MM_0_002_splits',
        # 'grid_search_MM_0_003_splits',
        # 'grid_search_MM_0_004_splits',
        # 'grid_search_MM_0_005_splits',
        # 'grid_search_MM_1_000_splits',
        # 'grid_search_MM_1_001_splits',
        # 'grid_search_MM_1_002_splits',
        # 'grid_search_MM_1_003_splits',
        # 'grid_search_MM_1_004_splits',
        # 'grid_search_MM_1_005_splits',
        # 'grid_search_MM_5_000_splits',
        # 'grid_search_MM_5_001_splits',
        # 'grid_search_MM_5_002_splits',
        # 'grid_search_MM_6_000_splits',
        # 'grid_search_MM_6_001_splits',
        # 'grid_search_MM_6_002_splits',
        # 'grid_search_MM_6_003_splits',
        # 'grid_search_MM_6_004_splits',
        # 'grid_search_MM_round2_7_000_splits',
        # 'grid_search_MM_round2_7_001_splits',
        # 'grid_search_MM_round2_7_002_splits',
        # 'grid_search_MM_round2_7_003_splits',
        # 'grid_search_MM_round2_7_004_splits',
        # 'grid_search_MM_round2_7_005_splits',
        # 'grid_search_MM_round2_7_006_splits'
        
        
        # 'grid_search_MM_30-10-34_0_000_splits',
        # 'grid_search_MM_30-10-34_0_001_splits',
        # 'grid_search_MM_30-10-34_0_002_splits',
        # 'grid_search_MM_30-10-34_0_003_splits',
        # 'grid_search_MM_30-10-34_0_004_splits',
        # 'grid_search_MM_30-10-34_0_005_splits', #!
        # 'grid_search_MM_30-10-34_0_006_splits',
        # 'grid_search_MM_30-10-34_0_007_splits',
        # 'grid_search_MM_30-10-34_0_008_splits',
        # 'grid_search_MM_30-10-34_0_009_splits',
        
        # 'grid_search_MM_late_30-10-34_1_000_splits',
        # 'grid_search_MM_late_30-10-34_1_001_splits',
        # 'grid_search_MM_late_30-10-34_1_002_splits',
        # 'grid_search_MM_late_30-10-34_1_003_splits',
        # 'grid_search_MM_late_30-10-34_1_004_splits', #!
        # 'grid_search_MM_late_30-10-34_1_005_splits',
        # 'grid_search_MM_late_30-10-34_1_006_splits',
        # 'grid_search_MM_late_30-10-34_1_007_splits',
        # 'grid_search_MM_late_30-10-34_1_008_splits',
        # 'grid_search_MM_late_30-10-34_1_009_splits',
        
        # 'grid_search_MM_01-17-32_0_000_splits',
        # 'grid_search_MM_01-17-32_0_001_splits',
        # 'grid_search_MM_01-17-32_0_002_splits', #!!
        # 'grid_search_MM_01-17-32_0_003_splits',
        # 'grid_search_MM_01-17-32_0_004_splits',
        # 'grid_search_MM_01-17-32_0_005_splits',
        # 'grid_search_MM_01-17-32_0_006_splits',
        # 'grid_search_MM_01-17-32_0_007_splits',
        # 'grid_search_MM_01-17-32_0_008_splits',
        # 'grid_search_MM_01-17-32_0_009_splits',
        # 'multiscale_splits_splits',
        # 'multiscale2_splits',
        # 'multiscale_simul_splits_splits',
        # 'Hierarchical_splits',
        
        # 'grid_search_hierarchical_06-09-51_0_000_splits',
        # 'grid_search_hierarchical_06-09-51_0_001_splits',
        # 'grid_search_hierarchical_06-09-51_0_002_splits',
        # 'grid_search_hierarchical_06-09-51_0_003_splits',
        # 'grid_search_hierarchical_06-09-51_0_004_splits',
        
        # 'gsMM_01-17-32-002_my_features_splits',
        # # 'gsMM_01-17-32-002_aug_features_splits',
        # 'grid_search_MM_30-10-34_0_005_splits',
        # 'gsMM_30-10-34-005_redo_splits',
        # 'gsMM_30-10-34-005_att_supervision0_splits',
        # 'gsMM_30-10-34-005_att_supervision1_splits',
        # 'gsMM_30-10-34-005_att_supervision10_splits',
        # 'gsMM_30-10-34-005_att_supervision_splits',
        # 'gsMM_30-10-34-005_my_features_splits',
        # 'gsMM_30-10-34-005_redo_my_features2_splits',
        # 'gsMM_30-10-34-005_aug_splits',
        # 'Hierarchical123_splits',
        # 'Hierarchical123winner_default_splits',
        # 'Hierarchical123winner_auxloss_splits',
        # 'Hierarchical123winner_splits',
        
        # 'Hierarchical123winner_redo_splits',
        # 'Hierarchicalvacation_winner_new_splits',
        # 'Hierarchicalvacation_winner_new_att_supervision1_splits',
        # 'Hierarchicalvacation_winner_new_att_supervision5_splits',
        # 'grid_search_hierarchical_vacation_redo_before_vacation2_splits',
        
        
        
        # # 'grid_search_hierarchical_07-13-52_0_000_splits',
        # # 'grid_search_hierarchical_07-13-52_0_001_splits',
        # # 'grid_search_hierarchical_09-12-11_0_000_splits',
        # # 'grid_search_hierarchical_09-12-11_0_001_splits'
        # # 'grid_search_hierarchical_vacation_18-12-52_1_011_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_031_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_009_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_017_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_030_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_010_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_020_splits',
        # # 'grid_search_hierarchical_vacation_18-12-49_0_023_splits',
        # # 'grid_search_hierarchical_vacation_18-12-52_1_002_splits',
        # 'grid_search_hierarchical_vacation_18-12-49_0_026_splits',
        # # 'attn_supervision_01-21-43_0_032_splits',
        # 'attn_supervision_01-21-43_0_030_splits',
        # # 'simultaneous_attn_supervision_01-21-42_gpu1_023_splits',
        # # 'simultaneous_attn_supervision_01-21-42_gpu1_000_splits',
        # 'simultaneous_attn_supervision_01-21-42_gpu1_009_splits',#!
        # # 'Random_seed2_splits',
        # # 'Random_seed3_splits',
        # # 'Random_seed4_splits',
        # # 'Random_seed5_splits',
        # 'advanced_hierarchical_tuning_12-16-32_0_017_splits',
        # 'advanced_hierarchical_tuning_12-16-32_0_021_splits',
        # 'advanced_simultaneous_tuning_12-09-01_2_002_splits',
        # 'attn_supervision_01-21-43_0_001_splits',
        # # 'attn_supervision_01-21-43_0_000_splits',
        'Random_splits',
        # # 'Random_steady_splits',
        # # 'Random_033_splits',
        # # 'simultaneous_attn_supervision_01-21-42_gpu1_015_splits',
        # # 'simultaneous_attn_supervision_01-21-42_gpu1_039_splits',
        # # 'simultaneous_attn_supervision_01-21-42_gpu1_040_splits'
        'advanced_hierarchical_tuning_12-16-32_0_017_splits',
        'advanced_hierarchical_tuning_12-16-32_0_021_splits',
        'grid_search_hierarchical_vacation_18-12-49_0_026_splits',
        'Ensemble_splits'
        # 'grid_search_MM_30-10-34_0_005_splits',
        
        # 'advanced_hierarchical_17_rerun_08-19-09_splits_0.1_splits',
        # 'advanced_hierarchical_21_rerun_08-19-09_splits_0.1_splits',
        # 'grid_search_hierarchical_rerun_08-19-09_000_splits_0.1_splits',
        # 'gsMM_30-10-34-005_rerun_08-19-09_splits_0.1_splits',
        
        # 'advanced_hierarchical_17_rerun_08-19-11_splits_0.2_splits',
        # 'advanced_hierarchical_21_rerun_08-19-11_splits_0.2_splits',
        # 'grid_search_hierarchical_rerun_08-19-11_000_splits_0.2_splits',
        # 'gsMM_30-10-34-005_rerun_08-19-11_splits_0.2_splits',
        

        
        
        
        
        
        
        
    ])
    
    