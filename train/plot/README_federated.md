# Federated Learning Plotting Script

This script has been updated to handle federated learning experiments with support for visualizing both client training progress and global model performance.

## New Features

### Federated Learning Structure Support
- **Client Training Logs**: `client_{id}_round_{round}` directories contain training metrics for individual clients per round
- **Server Evaluation Logs**: `client_server_{round}` directories contain global model evaluation results
  - `client_server_0`: Initial global model performance (before training)
  - `client_server_1`: Global model performance after round 1
  - `client_server_2`: Global model performance after round 2, etc.

### Metrics Format
The script now handles metrics in the format: `{metric}/{kind}/{submodel}`
- **metric**: Loss, Binary_Accuracy, F1, ROC_AUC, Accuracy
- **kind**: train, val, test
- **submodel**: MM, CLAM, CD

## Usage

### Single Federated Experiment
```bash
# Plot single federated experiment with MM submodel
python plot_run.py --federated --name test2_2_client_s1 --submodel MM

# Plot with all submodels
python plot_run.py --federated --name test2_2_client_s1 --submodel ALL

# Plot specific submodel
python plot_run.py --federated --name test2_2_client_s1 --submodel CLAM
```

### Compare Multiple Federated Experiments
```bash
# Compare test metrics across experiments
python plot_run.py --compare --names test2_1_client_s1 test2_2_client_s1 test2_s1 --submodel MM --metric_filter test

# Compare validation metrics
python plot_run.py --compare --names exp1 exp2 exp3 --submodel MM --metric_filter val

# Compare all metrics
python plot_run.py --compare --names exp1 exp2 --submodel MM --metric_filter all
```

### Auto-Detection
The script automatically detects federated experiments and switches to federated mode for experiment names containing "test2" or when using the `--federated` flag.

## Visualization Features

### Individual Experiment Plot
- **Client Training Curves**: Shows training/validation progress for each client across rounds
- **Global Model Performance**: Red squares with dotted lines show the global model evaluation after each round
- **Round Boundaries**: Vertical dashed lines separate different federated rounds
- **Submodel Support**: Can display MM, CLAM, CD, or all submodels simultaneously

### Comparison Plot
- **Global Model Trends**: Compares how the global model performance evolves across rounds for different experiments
- **Multiple Metrics**: Can filter by train/val/test or show all metrics
- **Clear Legend**: Distinguishes between different experiments with colors

## Output Files
- Single experiment: `{experiment_name}/federated_plot_{submodel}.png`
- Comparison: `federated_comparison_{submodel}_{metric_filter}.png`

## Example Commands

```bash
# Quick single experiment plot
python plot_run.py --name test2_2_client_s1

# Detailed federated plot with all submodels
python plot_run.py --federated --name test2_2_client_s1 --submodel ALL

# Compare global model performance across experiments
python plot_run.py --compare --names test2_1_client_s1 test2_2_client_s1 --metric_filter test
```

The script maintains backward compatibility with the original centralized training visualization functions.