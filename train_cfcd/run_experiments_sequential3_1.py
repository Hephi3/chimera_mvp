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
# GPU 3, Script 1: Focus on federated_train sampling and repr method_global
EXPERIMENTS = [
    # PRIORITY 1-3: Sampling (federated_train) - exploring edge cases and combinations
    
    # Exp 1: MS with 1 sample, low temperature, high variance
    (federated_train, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "73_MS_1_05_08",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "num_sampled": 1,
        "temperature": 0.5,
        "variance_scale": 0.8,
    }, "MS: num_sampled=1, temp=0.5, var=0.8"),
    
    # Exp 2: MS with 4 samples, low temperature, high variance
    (federated_train, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "73_MS_4_05_08",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "num_sampled": 4,
        "temperature": 0.5,
        "variance_scale": 0.8,
    }, "MS: num_sampled=4, temp=0.5, var=0.8"),
    
    # Exp 3: MS with 2 samples, very high temperature, medium variance
    (federated_train, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "73_MS_2_3_06",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "num_sampled": 2,
        "temperature": 3.0,
        "variance_scale": 0.6,
    }, "MS: num_sampled=2, temp=3.0, var=0.6"),
    
    # PRIORITY 4-7: repr Method Global - exploring adaptation rates and temperatures
    
    # Exp 4: repr MG with high adaptation, medium temperature
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MG_05_02t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.5,
        "temperature": 0.2,
    }, "repr MG: adaptation_rate=0.5, temperature=0.2"),
    
    # Exp 5: repr MG with low adaptation, very low temperature
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MG_01_005t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.1,
        "temperature": 0.05,
    }, "repr MG: adaptation_rate=0.1, temperature=0.05"),
    
    # Exp 6: repr MG with medium adaptation, low temperature
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MG_03_01t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.3,
        "temperature": 0.1,
    }, "repr MG: adaptation_rate=0.3, temperature=0.1"),
    
    # Exp 7: repr MG with medium adaptation, high temperature
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MG_02_05t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.2,
        "temperature": 0.5,
    }, "repr MG: adaptation_rate=0.2, temperature=0.5"),
    
    # PRIORITY 8-9: Combined methods
    
    # Exp 8: Combined ML+MG on federated_train
    (federated_train, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "73_MLMG_05_3str_03_02t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.5,
        "strictness": 3.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.3,
        "temperature": 0.2,
    }, "MLMG: client=0.5/str=3, server=0.3/temp=0.2"),
    
    # Exp 9: Combined ML+MS on federated_train
    (federated_train, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "73_MLMS_02_2str_3s_15_04",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.2,
        "strictness": 2.0,
        "num_sampled": 3,
        "temperature": 1.5,
        "variance_scale": 0.4,
    }, "MLMS: ML(0.2/2) + MS(3/1.5/0.4)"),
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
