# Experiment Runners

This directory contains scripts and configurations for running systematic experiments with different hyperparameter combinations.

## Available Scripts

### 1. `experiment_runner.py` - Command-line Parameter Sweeps

A flexible command-line tool for running parameter sweeps without modifying your original code.

#### Basic Usage Examples:

```bash
# Test different numbers of clients
python experiment_runner.py --base_exp_code client_test --num_clients_list 2 3 5 --num_rounds 5

# Test different numbers of rounds
python experiment_runner.py --base_exp_code rounds_test --num_rounds_list 3 5 10 --num_clients 3

# Test different splits
python experiment_runner.py --base_exp_code split_test --split_dir_list chimera_3_0.1 chimera_5_0.1 chimera_10_0.1

# Multi-seed runs for statistical significance
python experiment_runner.py --base_exp_code multi_seed --seeds 1 2 3 4 5 --num_clients 3 --num_rounds 5

# Complex multi-parameter sweep
python experiment_runner.py --base_exp_code complex --num_clients_list 2 3 --num_rounds_list 3 5 --lr_list 1e-4 5e-4
```

#### Advanced Options:

```bash
# Dry run to see what commands would be executed
python experiment_runner.py --base_exp_code test --num_clients_list 2 3 --dry_run

# Log all output to a file
python experiment_runner.py --base_exp_code test --num_clients_list 2 3 --log_file logs/experiment.log

# Continue running other experiments if one fails
python experiment_runner.py --base_exp_code test --num_clients_list 2 3 --continue_on_error
```

### 2. `config_experiment_runner.py` - Configuration-based Experiments

Uses YAML configuration files for more complex and reproducible experiment setups.

#### Basic Usage:

```bash
# Run experiments from a config file
python config_experiment_runner.py --config experiments/client_sweep.yaml

# Dry run to see what would be executed
python config_experiment_runner.py --config experiments/complex_sweep.yaml --dry_run

# Continue on errors and log to custom directory
python config_experiment_runner.py --config experiments/multi_seed.yaml --continue_on_error --log_dir my_logs
```

## Available Experiment Configurations

### Pre-configured Experiments:

1. **`client_sweep.yaml`** - Tests performance with 2, 3, 5, 8, 10 clients
2. **`rounds_sweep.yaml`** - Tests with 3, 5, 10, 15, 20 federated learning rounds
3. **`split_comparison.yaml`** - Compares different data split configurations
4. **`multi_seed.yaml`** - Runs with 10 different seeds for statistical validation
5. **`hyperparameter_sweep.yaml`** - Tests different learning rates and bag weights
6. **`complex_sweep.yaml`** - Multi-dimensional parameter sweep

### Creating Custom Configurations:

Create a new YAML file in the `experiments/` directory:

```yaml
# my_experiment.yaml
experiment_name: "my_custom_experiment"

script_config:
  script: "federated_train.py"

base_parameters:
  gpus: [1]
  num_rounds: 5
  split_dir: "chimera_3_0.1"
  seed: 1
  no_verbose: true
  max_epochs: 100

sweep_parameters:
  num_clients: [2, 3, 5]
  lr: [1e-4, 5e-4, 1e-3]
  bag_weight: [0.7, 0.8, 0.9]
```

This will generate all combinations of the sweep parameters (3 × 3 × 3 = 27 experiments).

## Output and Logging

- **Experiment names**: Automatically generated based on parameters (e.g., `client_test_c3_r5_s1`)
- **Log files**: Timestamped logs in `experiment_logs/` directory (configurable)
- **Results**: Standard federated learning results in your configured results directory
- **Progress tracking**: Real-time progress with timing information

## Tips for Large Experiments

1. **Use dry run first**: Always test with `--dry_run` to verify your experiment setup
2. **Start small**: Test with a subset of parameters before running large sweeps
3. **Use continue_on_error**: For long-running sweeps where individual failures shouldn't stop the entire run
4. **Monitor resources**: Large parameter sweeps can take significant time and computational resources
5. **Check logs**: Monitor the log files to track progress and catch issues early

## Example Workflow

```bash
# 1. Test your configuration
python config_experiment_runner.py --config experiments/client_sweep.yaml --dry_run

# 2. Run a small test first
python experiment_runner.py --base_exp_code quick_test --num_clients_list 2 3 --num_rounds 2 --max_epochs_list 10

# 3. Run your main experiment
python config_experiment_runner.py --config experiments/client_sweep.yaml --continue_on_error

# 4. Check results and logs
ls -la experiment_logs/
ls -la results/
```

## Integration with TensorBoard

All experiments will automatically log to TensorBoard. After running experiments, you can view all results:

```bash
tensorboard --logdir results/
```

This will show all your experiments side by side for easy comparison.