#!/usr/bin/env python3
"""
Analyze and visualize prototype weights from training.
Shows average weights per datapoint and how weights change over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def parse_weights_file(filepath):
    """Parse the prototype weights file and organize by datapoint."""
    weights_by_datapoint = defaultdict(list)
    datapoint_order = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':')
            if len(parts) != 2:
                continue
                
            datapoint_id = parts[0]
            try:
                weight = float(parts[1])
            except ValueError:
                continue
            
            # Track order of first appearance
            if datapoint_id not in weights_by_datapoint:
                datapoint_order.append(datapoint_id)
            
            weights_by_datapoint[datapoint_id].append(weight)
    
    return weights_by_datapoint, datapoint_order

def plot_weight_analysis(weights_by_datapoint, datapoint_order, output_prefix):
    """Create visualizations for weight analysis."""
    
    # Calculate statistics
    avg_weights = {dp: np.mean(weights) for dp, weights in weights_by_datapoint.items()}
    std_weights = {dp: np.std(weights) for dp, weights in weights_by_datapoint.items()}
    
    # Sort datapoints by average weight
    sorted_datapoints = sorted(datapoint_order, key=lambda x: avg_weights[x])
    
    # Figure 1: Average weights per datapoint (bar plot)
    fig, ax = plt.subplots(figsize=(20, 8))
    x_pos = np.arange(len(sorted_datapoints))
    avgs = [avg_weights[dp] for dp in sorted_datapoints]
    stds = [std_weights[dp] for dp in sorted_datapoints]
    
    ax.bar(x_pos, avgs, yerr=stds, alpha=0.7, capsize=3)
    ax.set_xlabel('Datapoint', fontsize=12)
    ax.set_ylabel('Average Weight', fontsize=12)
    ax.set_title('Average Prototype Weights per Datapoint (sorted by weight)', fontsize=14)
    ax.set_xticks(x_pos[::max(1, len(sorted_datapoints)//50)])
    ax.set_xticklabels([sorted_datapoints[i] for i in range(0, len(sorted_datapoints), max(1, len(sorted_datapoints)//50))], 
                       rotation=45, ha='right', fontsize=8)
    ax.axhline(y=np.mean(avgs), color='r', linestyle='--', label=f'Overall mean: {np.mean(avgs):.3f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_avg_weights_per_datapoint.png', dpi=150)
    print(f"Saved: {output_prefix}_avg_weights_per_datapoint.png")
    plt.close()
    
    # Figure 2: Weight evolution over time for all datapoints
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot each datapoint's weight evolution
    for dp in datapoint_order[:30]:  # Plot first 30 for visibility
        weights = weights_by_datapoint[dp]
        ax.plot(range(len(weights)), weights, alpha=0.6, linewidth=1, label=dp)
    
    ax.set_xlabel('Epoch/Iteration', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Weight Evolution Over Time (first 30 datapoints)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_weight_evolution_sample.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_weight_evolution_sample.png")
    plt.close()
    
    # Figure 3: Heatmap of weights over time
    # Select subset of datapoints for readability
    num_display = min(50, len(datapoint_order))
    display_datapoints = sorted_datapoints[-num_display:]  # Top weighted datapoints
    
    # Create matrix
    max_iterations = max(len(weights_by_datapoint[dp]) for dp in display_datapoints)
    weight_matrix = np.full((num_display, max_iterations), np.nan)
    
    for i, dp in enumerate(display_datapoints):
        weights = weights_by_datapoint[dp]
        weight_matrix[i, :len(weights)] = weights
    
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(weight_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.set_xlabel('Epoch/Iteration', fontsize=12)
    ax.set_ylabel('Datapoint', fontsize=12)
    ax.set_title(f'Weight Heatmap Over Time (Top {num_display} weighted datapoints)', fontsize=14)
    ax.set_yticks(range(num_display))
    ax.set_yticklabels(display_datapoints, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_weight_heatmap.png', dpi=150)
    print(f"Saved: {output_prefix}_weight_heatmap.png")
    plt.close()
    
    # Figure 4: Distribution of average weights
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(list(avg_weights.values()), bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average Weight', fontsize=12)
    ax.set_ylabel('Number of Datapoints', fontsize=12)
    ax.set_title('Distribution of Average Weights Across Datapoints', fontsize=14)
    ax.axvline(x=np.mean(list(avg_weights.values())), color='r', linestyle='--', 
               label=f'Mean: {np.mean(list(avg_weights.values())):.3f}')
    ax.axvline(x=np.median(list(avg_weights.values())), color='g', linestyle='--', 
               label=f'Median: {np.median(list(avg_weights.values())):.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_weight_distribution.png', dpi=150)
    print(f"Saved: {output_prefix}_weight_distribution.png")
    plt.close()
    
    # Print statistics
    print("\n=== Weight Statistics ===")
    print(f"Total datapoints: {len(weights_by_datapoint)}")
    print(f"Average weight across all: {np.mean(list(avg_weights.values())):.4f}")
    print(f"Std dev of average weights: {np.std(list(avg_weights.values())):.4f}")
    print(f"Min average weight: {min(avg_weights.values()):.4f}")
    print(f"Max average weight: {max(avg_weights.values()):.4f}")
    
    print("\n=== Top 10 Highest Weighted Datapoints ===")
    for dp in sorted_datapoints[-10:]:
        print(f"{dp}: {avg_weights[dp]:.4f} ± {std_weights[dp]:.4f} (n={len(weights_by_datapoint[dp])})")
    
    print("\n=== Top 10 Lowest Weighted Datapoints ===")
    for dp in sorted_datapoints[:10]:
        print(f"{dp}: {avg_weights[dp]:.4f} ± {std_weights[dp]:.4f} (n={len(weights_by_datapoint[dp])})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        weights_file = sys.argv[1]
    else:
        weights_file = "results/73_ML_02_str1_sp1_s3/Fold0/prototype_weights.txt"
    
    output_prefix = weights_file.replace('.txt', '')
    
    print(f"Analyzing weights from: {weights_file}")
    weights_by_datapoint, datapoint_order = parse_weights_file(weights_file)
    
    if not weights_by_datapoint:
        print("No valid weight data found in file!")
        sys.exit(1)
    
    plot_weight_analysis(weights_by_datapoint, datapoint_order, output_prefix)
    print("\nAnalysis complete!")
