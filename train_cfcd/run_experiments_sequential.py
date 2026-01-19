#!/usr/bin/env python3
"""
Sequential experiment runner for federated learning experiments.
Runs multiple experiments one after another by directly calling run_experiment.
"""

import time
import datetime
import sys
import os
import copy

# Import the training scripts
import federated_train
import federated_train_repr
from hyperparameters import get_hyperparameters

# Define your experiments here
# Each experiment is a tuple of (module, parameters_dict, description)
EXPERIMENTS = [
    # Example experiments - modify these according to your needs
    
    # Experiment 1: federated_train with baseline settings
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "baseline_exp1",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "seed": 1,
    }, "Baseline federated training experiment 1"),
    
    # Experiment 2: federated_train_repr with prototype method
    (federated_train_repr, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "repr_exp1",
        "verbose": False,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.8,
        "strictness": 1.0,
        "seed": 1,
        "method_global": True,
        "proto_adaptation_rate_server": 0.8,
        "temperature": 1.0,
    }, "Prototype representation experiment 1"),
    
    # Experiment 3: federated_train_repr with sampling
    (federated_train_repr, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "repr_sample_exp1",
        "verbose": False,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "num_sampled": 2,
        "temperature": 2,
        "variance_scale": 0.4,
        "proto_adaptation_rate_client": 0.8,
        "strictness": 1.0,
        "num_rounds": 40,
        "seed": 1,
    }, "Prototype with sampling experiment 1"),
    
    # Experiment 4: Different seed
    (federated_train_repr, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "repr_exp2_seed2",
        "verbose": False,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.8,
        "strictness": 1.0,
        "num_rounds": 40,
        "seed": 2,
    }, "Prototype experiment with seed 2"),
    
    # Add more experiments as needed...
]


def run_experiment(module, params, description, exp_num, total_exps):
    """Run a single experiment by directly calling the module's run_experiment function."""
    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_num}/{total_exps}: {description}")
    print("="*80)
    print(f"Module: {module.__name__}")
    print(f"Parameters: {params}")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    import time
    start_time_all = time.time()
    
    try:
        # Get hyperparameters and update with custom params
        orig_args = get_hyperparameters()
        
        # Update args with provided parameters
        for key, value in params.items():
            setattr(orig_args, key, value)
            
        import time
        pairs = [(i + 1, 3) for i in range(5)]
        pairs = [(i + 1, s) for i in range(5) for s in [1,2]]
        import copy
        for split, seed in pairs:
            start_time = time.time()
            print("Split {}, seed {}".format(split, seed))
            args = copy.deepcopy(orig_args)
            args.seed = seed
            args.split_dir = f'{orig_args.split_dir}_{split}'
            args.exp_code = f'{orig_args.exp_code}_sp{split}'
            module.run_experiment(args)
            end_time = time.time()
            print(f"Experiment completed in {end_time - start_time:.2f} seconds")
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print(f"✓ Experiment {exp_num} completed successfully!")
        print(f"Duration: {elapsed_time/60:.2f} minutes ({elapsed_time:.0f} seconds)")
        print(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        return True, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time_all
        print("\n" + "="*80)
        print(f"✗ Experiment {exp_num} encountered an error!")
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"Duration before error: {elapsed_time/60:.2f} minutes")
        print("="*80 + "\n")
        
        return False, elapsed_time
    
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("Experiment interrupted by user (Ctrl+C)")
        print("="*80)
        raise


def main():
    """Main function to run all experiments sequentially."""
    print("\n" + "#"*80)
    print("SEQUENTIAL EXPERIMENT RUNNER")
    print("#"*80)
    print(f"Total experiments to run: {len(EXPERIMENTS)}")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*80 + "\n")
    
    total_start_time = time.time()
    results = []
    
    for i, (script, params, desc) in enumerate(EXPERIMENTS, 1):
        success, duration = run_experiment(script, params, desc, i, len(EXPERIMENTS))
        results.append({
            'exp_num': i,
            'script': script,
            'description': desc,
            'success': success,
            'duration': duration
        })
        
        # Optional: Add a short pause between experiments
        if i < len(EXPERIMENTS):
            print(f"Waiting 10 seconds before next experiment...\n")
            time.sleep(10)
    
    # Print summary
    total_elapsed = time.time() - total_start_time
    print("\n" + "#"*80)
    print("EXPERIMENT SUMMARY")
    print("#"*80)
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.2f} minutes)")
    print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    print("-"*80)
    
    successful = 0
    failed = 0
    
    for res in results:
        status = "✓ SUCCESS" if res['success'] else "✗ FAILED"
        print(f"{res['exp_num']:2d}. [{status}] {res['description']}")
        print(f"    Script: {res['script']}")
        print(f"    Duration: {res['duration']/60:.2f} minutes")
        
        if res['success']:
            successful += 1
        else:
            failed += 1
    
    print("-"*80)
    print(f"Successful: {successful}/{len(EXPERIMENTS)}")
    print(f"Failed: {failed}/{len(EXPERIMENTS)}")
    print("#"*80 + "\n")
    
    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment runner stopped by user.")
        sys.exit(130)
