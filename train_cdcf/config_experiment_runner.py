#!/usr/bin/env python3
"""
Configuration-based Experiment Runner

This script allows you to define complex experiment configurations using YAML files
and run them systematically.

Usage:
    python config_experiment_runner.py --config experiments/client_sweep.yaml
    python config_experiment_runner.py --config experiments/complex_sweep.yaml --dry_run
"""

import os
import sys
import yaml
import json
import argparse
import itertools
import subprocess
import time
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments from configuration files')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment configuration file (YAML or JSON)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show commands without executing them')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue if an experiment fails')
    parser.add_argument('--log_dir', type=str, default='experiment_logs',
                        help='Directory to store experiment logs')
    return parser.parse_args()

def load_config(config_path):
    """Load experiment configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def expand_parameter_combinations(sweep_config):
    """Expand parameter sweep configuration into all combinations."""
    
    # Extract base parameters and sweep parameters
    base_params = sweep_config.get('base_parameters', {})
    sweep_params = sweep_config.get('sweep_parameters', {})
    
    if not sweep_params:
        # If no sweep parameters, return just the base parameters
        return [base_params]
    
    # Generate all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    combinations = []
    for combination in itertools.product(*param_values):
        # Start with base parameters
        combo_dict = base_params.copy()
        # Add sweep parameter combination
        combo_dict.update(dict(zip(param_names, combination)))
        combinations.append(combo_dict)
    
    return combinations

def build_command_from_config(params, script_config):
    """Build command from parameters and script configuration."""
    
    script_path = script_config.get('script', 'federated_train.py')
    
    # Start with base command
    cmd = ['python', script_path]
    
    # Add GPU configuration
    if 'gpus' in params:
        gpus = params['gpus']
        if isinstance(gpus, int):
            gpus = [gpus]
        cmd.extend(['--gpus'] + [str(gpu) for gpu in gpus])
    
    # Add all other parameters
    for key, value in params.items():
        if key == 'gpus':  # Already handled
            continue
            
        # Convert parameter name to command line format
        param_name = f'--{key}'
        
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(param_name)
        elif isinstance(value, list):
            cmd.extend([param_name] + [str(v) for v in value])
        else:
            cmd.extend([param_name, str(value)])
    
    return cmd

def generate_experiment_name(params, base_name):
    """Generate a unique experiment name based on parameters."""
    
    name_parts = [base_name]
    
    # Add key distinguishing parameters to the name
    key_params = ['num_clients', 'num_rounds', 'seed', 'split_dir', 'lr']
    
    for param in key_params:
        if param in params:
            value = params[param]
            if param == 'split_dir':
                # Use basename for split_dir
                value = os.path.basename(str(value))
            name_parts.append(f"{param}_{value}")
    
    return "_".join(name_parts)

def log_message(message, log_file=None):
    """Log message to console and optionally to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted_message + '\n')

def run_experiment(cmd, params, log_file=None, continue_on_error=False):
    """Run a single experiment."""
    
    exp_name = params.get('exp_code', 'unnamed')
    log_message(f"Starting experiment: {exp_name}", log_file)
    log_message(f"Command: {' '.join(cmd)}", log_file)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        duration = end_time - start_time
        
        log_message(f"✓ Experiment completed successfully in {duration:.2f}s: {exp_name}", log_file)
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        log_message(f"✗ Experiment failed after {duration:.2f}s: {exp_name}", log_file)
        log_message(f"Error code: {e.returncode}", log_file)
        
        if e.stderr:
            log_message("STDERR:", log_file)
            log_message(e.stderr, log_file)
        
        if not continue_on_error:
            raise
        
        return False

def main():
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    config_name = Path(args.config).stem
    log_file = log_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_message(f"Starting experiment run from config: {args.config}", log_file)
    log_message(f"Logging to: {log_file}", log_file)
    
    # Process experiment configuration
    experiment_name = config.get('experiment_name', 'unnamed_experiment')
    script_config = config.get('script_config', {})  
    # Generate parameter combinations
    combinations = expand_parameter_combinations(config)
    
    # Generate experiment names and update exp_code
    for combo in combinations:
        if 'exp_code' not in combo:
            combo['exp_code'] = generate_experiment_name(combo, experiment_name)
    
    log_message(f"Generated {len(combinations)} experiment combinations", log_file)
    
    # Show all experiments
    for i, combo in enumerate(combinations, 1):
        cmd = build_command_from_config(combo, script_config)
        log_message(f"Experiment {i}: {combo['exp_code']}", log_file)
        log_message(f"  Command: {' '.join(cmd)}", log_file)
    
    if args.dry_run:
        log_message("Dry run complete - no experiments were executed", log_file)
        return
    
    # Run experiments
    successful = 0
    failed = 0
    total_start_time = time.time()
    
    for i, combo in enumerate(combinations, 1):
        log_message(f"\n{'='*60}", log_file)
        log_message(f"Running experiment {i}/{len(combinations)}", log_file)
        
        cmd = build_command_from_config(combo, script_config)
        
        try:
            if run_experiment(cmd, combo, log_file, args.continue_on_error):
                successful += 1
            else:
                failed += 1
        except subprocess.CalledProcessError:
            failed += 1
            if not args.continue_on_error:
                break
        except KeyboardInterrupt:
            log_message("\nExperiments interrupted by user", log_file)
            break
    
    # Summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    log_message(f"\n{'='*60}", log_file)
    log_message(f"Experiment sweep completed in {total_duration:.2f}s", log_file)
    log_message(f"Successful: {successful}", log_file)
    log_message(f"Failed: {failed}", log_file)
    log_message(f"Total: {successful + failed}", log_file)

if __name__ == "__main__":
    main()