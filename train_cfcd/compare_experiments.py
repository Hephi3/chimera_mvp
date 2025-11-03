import tensorflow as tf
import os

def get_test_metrics(exp_name, fold=0):
    """Extract test metrics for both stages from an experiment"""
    log_dir = f"results/{exp_name}/Fold{fold}/log"
    
    metrics = {0: {}, 1: {}}  # stage 0 and stage 1
    
    for item in os.listdir(log_dir):
        if 'server' in item:
            round_num = int(item.split('_')[-1])
            event_dir = os.path.join(log_dir, item)
            
            for file in os.listdir(event_dir):
                if file.startswith('events.out.tfevents'):
                    event_file = os.path.join(event_dir, file)
                    
                    for event in tf.compat.v1.train.summary_iterator(event_file):
                        for value in event.summary.value:
                            if '/test/MM/' in value.tag:
                                # Extract stage from metric name
                                if value.tag.endswith('/0'):
                                    stage = 0
                                elif value.tag.endswith('/1'):
                                    stage = 1
                                else:
                                    continue
                                
                                metric_name = value.tag.split('/')[0]
                                
                                if round_num not in metrics[stage]:
                                    metrics[stage][round_num] = {}
                                    
                                metrics[stage][round_num][metric_name] = value.simple_value
    
    return metrics

# Compare del1 and del2
print("Comparing Stage 0 (non-augmented) metrics:")
print("=" * 60)

try:
    del1_metrics = get_test_metrics('del1', fold=0)
    del2_metrics = get_test_metrics('del2', fold=0)
    
    # Compare Stage 0
    for round_num in sorted(del1_metrics[0].keys()):
        if round_num in del2_metrics[0]:
            print(f"\nRound {round_num}:")
            for metric in ['Binary_Accuracy', 'F1']:
                if metric in del1_metrics[0][round_num] and metric in del2_metrics[0][round_num]:
                    val1 = del1_metrics[0][round_num][metric]
                    val2 = del2_metrics[0][round_num][metric]
                    diff = abs(val1 - val2)
                    print(f"  {metric}: del1={val1:.4f}, del2={val2:.4f}, diff={diff:.6f}")
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure del1 and del2 experiments have been run!")
