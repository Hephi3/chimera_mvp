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
# GPU 2, Script 1: Focus on method_local variations (federated_train)
EXPERIMENTS = [
    # PRIORITY 1-3: Method Local - exploring adaptation rates and strictness combinations
    
    # Exp 1: ML with high adaptation rate, medium strictness
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_ML_05_str3",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.5,
        "strictness": 3.0,
    }, "ML: adaptation_rate=0.5, strictness=3"),
    
    # Exp 2: ML with low adaptation rate, high strictness
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_ML_01_str5",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.1,
        "strictness": 5.0,
    }, "ML: adaptation_rate=0.1, strictness=5"),
    
    # Exp 3: ML with medium adaptation rate, low strictness
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_ML_03_str05",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.3,
        "strictness": 0.5,
    }, "ML: adaptation_rate=0.3, strictness=0.5"),
    
    # PRIORITY 4-6: Method Global - exploring adaptation rates and temperatures
    
    # Exp 4: MG with high adaptation, low temperature
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_MG_05_01t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.5,
        "temperature": 0.1,
    }, "MG: adaptation_rate=0.5, temperature=0.1"),
    
    # Exp 5: MG with medium adaptation, high temperature
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_MG_03_05t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.3,
        "temperature": 0.5,
    }, "MG: adaptation_rate=0.3, temperature=0.5"),
    
    # Exp 6: MG with low adaptation, very low temperature
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_MG_01_002t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_global": True,
        "proto_adaptation_rate_server": 0.1,
        "temperature": 0.02,
    }, "MG: adaptation_rate=0.1, temperature=0.02"),
    
    # PRIORITY 7-8: Sampling variations
    
    # Exp 7: Sampling with 3 samples, low temperature, low variance
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_MS_3_05_02",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "num_sampled": 3,
        "temperature": 0.5,
        "variance_scale": 0.2,
    }, "MS: num_sampled=3, temp=0.5, var=0.2"),
    
    # Exp 8: Sampling with 4 samples, medium temperature, medium variance
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_MS_4_15_06",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "num_sampled": 4,
        "temperature": 1.5,
        "variance_scale": 0.6,
    }, "MS: num_sampled=4, temp=1.5, var=0.6"),
    
    # PRIORITY 9: Combined methods
    
    # Exp 9: ML + MG combination with balanced params
    (federated_train, {
        "gpus": [2],
        "num_clients": 3,
        "exp_code": "73_MLMG_03_2str_05_01t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.3,
        "strictness": 2.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.5,
        "temperature": 0.1,
    }, "MLMG: client=0.3/str=2, server=0.5/temp=0.1"),
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
