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
# GPU 3, Script 2: Focus on repr combined methods and edge cases
EXPERIMENTS = [
    # PRIORITY 1-3: repr Combined ML+MG - exploring parameter interactions
    
    # Exp 1: repr MLMG with high client/low server, medium strictness
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MLMG_05_2str_01_01t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.5,
        "strictness": 2.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.1,
        "temperature": 0.1,
    }, "repr MLMG: client=0.5/str=2, server=0.1/temp=0.1"),
    
    # Exp 2: repr MLMG with balanced adaptation, high strictness
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MLMG_03_4str_03_02t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.3,
        "strictness": 4.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.3,
        "temperature": 0.2,
    }, "repr MLMG: client=0.3/str=4, server=0.3/temp=0.2"),
    
    # Exp 3: repr MLMG with low client/high server, low strictness
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MLMG_01_1str_05_005t",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.1,
        "strictness": 1.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.5,
        "temperature": 0.05,
    }, "repr MLMG: client=0.1/str=1, server=0.5/temp=0.05"),
    
    # PRIORITY 4-6: repr Combined ML+MS - testing local + sampling interaction
    
    # Exp 4: repr MLMS with medium client, low sampling variance
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MLMS_03_2str_2s_1_02",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.3,
        "strictness": 2.0,
        "num_sampled": 2,
        "temperature": 1.0,
        "variance_scale": 0.2,
    }, "repr MLMS: ML(0.3/2) + MS(2/1.0/0.2)"),
    
    # Exp 5: repr MLMS with high client, high sampling variance
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MLMS_05_1str_3s_2_08",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.5,
        "strictness": 1.0,
        "num_sampled": 3,
        "temperature": 2.0,
        "variance_scale": 0.8,
    }, "repr MLMS: ML(0.5/1) + MS(3/2.0/0.8)"),
    
    # Exp 6: repr MLMS with low client, many samples
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_MLMS_01_3str_4s_15_04",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.1,
        "strictness": 3.0,
        "num_sampled": 4,
        "temperature": 1.5,
        "variance_scale": 0.4,
    }, "repr MLMS: ML(0.1/3) + MS(4/1.5/0.4)"),
    
    # PRIORITY 7-9: repr All methods combined - testing full system
    
    # Exp 7: repr ALL with balanced parameters
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_ALL_03_2str_02_01t_2s_04",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.3,
        "strictness": 2.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.2,
        "temperature": 0.1,
        "num_sampled": 2,
        "variance_scale": 0.4,
    }, "repr ALL: ML(0.3/2) + MG(0.2/0.1) + MS(2/0.4)"),
    
    # Exp 8: repr ALL with aggressive parameters
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_ALL_05_1str_05_02t_3s_08",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.5,
        "strictness": 1.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.5,
        "temperature": 0.2,
        "num_sampled": 3,
        "variance_scale": 0.8,
    }, "repr ALL: ML(0.5/1) + MG(0.5/0.2) + MS(3/0.8)"),
    
    # Exp 9: repr ALL with conservative parameters
    (federated_train_repr, {
        "gpus": [3],
        "num_clients": 3,
        "exp_code": "repr_ALL_01_4str_01_005t_1s_02",
        "no_verbose": True,
        "split_dir": "chimera_3_5_2_0.2_0.7_0.3",
        "augmentations": ["0=features_1536_fixed", "1=features_1536_fixed", "2=Aug0_brightness_460"],
        "method_local": True,
        "proto_adaptation_rate_client": 0.1,
        "strictness": 4.0,
        "method_global": True,
        "proto_adaptation_rate_server": 0.1,
        "temperature": 0.05,
        "num_sampled": 1,
        "variance_scale": 0.2,
    }, "repr ALL: ML(0.1/4) + MG(0.1/0.05) + MS(1/0.2)"),
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
