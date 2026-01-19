"""
Script to plot prototype weights from the log file.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def parse_weights_file(filepath):
    """Parse the prototype_weights.txt file."""
    slide_ids = []
    weights = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                # Parse slide_id:weight format
                slide_id, weight = line.split(':', 1)
                slide_ids.append(slide_id.strip())
                weights.append(float(weight.strip()))
    
    # Create synthetic epoch/batch/label data since the file only contains weights
    num_weights = len(weights)
    
    return {
        'slide_ids': slide_ids,
        'epoch': np.zeros(num_weights, dtype=int),  # All same epoch
        'batch': np.arange(num_weights),  # Sequential batches
        'label': np.zeros(num_weights, dtype=int),  # Unknown labels
        'weight': np.array(weights),
        'loss_before': np.zeros(num_weights),  # Not available
        'loss_after': np.zeros(num_weights)  # Not available
    }


def plot_weights(data, output_dir=None):
    """Create comprehensive plots of prototype weights."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Weight distribution over time
    ax = axes[0, 0]
    ax.plot(range(len(data['weight'])), data['weight'], linewidth=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Weight')
    ax.set_title('Prototype Weight Over Samples')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Weight histogram
    ax = axes[0, 1]
    ax.hist(data['weight'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(data['weight']), color='red', linestyle='--', label=f'Mean: {np.mean(data["weight"]):.4f}')
    ax.axvline(np.median(data['weight']), color='green', linestyle='--', label=f'Median: {np.median(data["weight"]):.4f}')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative weight distribution
    ax = axes[1, 0]
    sorted_weights = np.sort(data['weight'])
    cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    ax.plot(sorted_weights, cumulative)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Weight Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Rolling average
    ax = axes[1, 1]
    window_size = min(100, len(data['weight']) // 10)
    if window_size > 1:
        rolling_mean = np.convolve(data['weight'], np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size-1, len(data['weight'])), rolling_mean)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'Rolling Mean (window={window_size})')
        ax.set_title('Weight Trend Over Time')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for rolling average', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'prototype_weights_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.show()


def plot_weight_statistics(data, output_dir=None):
    """Create statistical summary plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Weight percentiles
    ax = axes[0, 0]
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    percentile_values = [np.percentile(data['weight'], p) for p in percentiles]
    ax.bar(range(len(percentiles)), percentile_values, tick_label=[f'{p}%' for p in percentiles])
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Weight Value')
    ax.set_title('Weight Percentiles')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Weight bins
    ax = axes[0, 1]
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 1.0]
    bin_counts, _ = np.histogram(data['weight'], bins=bins)
    bin_labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
    ax.bar(range(len(bin_counts)), bin_counts, tick_label=bin_labels)
    ax.set_xlabel('Weight Range')
    ax.set_ylabel('Count')
    ax.set_title('Weight Distribution by Range')
    ax.grid(True, alpha=0.3, axis='y')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    # Plot 3: Box plot
    ax = axes[1, 0]
    ax.boxplot([data['weight']], labels=['All Weights'])
    ax.set_ylabel('Weight')
    ax.set_title('Weight Distribution (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Violin plot
    ax = axes[1, 1]
    parts = ax.violinplot([data['weight']], positions=[0], showmeans=True, showmedians=True)
    ax.set_xticks([0])
    ax.set_xticklabels(['All Weights'])
    ax.set_ylabel('Weight')
    ax.set_title('Weight Distribution (Violin Plot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'prototype_weights_statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.show()


def print_statistics(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("PROTOTYPE WEIGHTS STATISTICS")
    print("="*60)
    print(f"Total samples: {len(data['weight'])}")
    print(f"\nWeight statistics:")
    print(f"  Mean: {np.mean(data['weight']):.6f}")
    print(f"  Median: {np.median(data['weight']):.6f}")
    print(f"  Std: {np.std(data['weight']):.6f}")
    print(f"  Min: {np.min(data['weight']):.6f}")
    print(f"  Max: {np.max(data['weight']):.6f}")
    print(f"  25th percentile: {np.percentile(data['weight'], 25):.6f}")
    print(f"  75th percentile: {np.percentile(data['weight'], 75):.6f}")
    
    # Count weights in different ranges
    print(f"\nWeight ranges:")
    ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 1.0)]
    for low, high in ranges:
        count = np.sum((data['weight'] >= low) & (data['weight'] < high))
        pct = 100 * count / len(data['weight'])
        print(f"  [{low:.2f}, {high:.2f}): {count} ({pct:.1f}%)")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot prototype weights from log file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to prototype_weights.txt file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: same as input file)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not show plots interactively')
    
    args = parser.parse_args()
    
    # Parse data
    print(f"Reading data from: {args.input}")
    data = parse_weights_file(args.input)
    
    # Print statistics
    print_statistics(data)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(args.input).parent
    
    # Create plots
    print("Creating plots...")
    
    if args.no_show:
        plt.ioff()
    
    plot_weights(data, args.output_dir)
    plot_weight_statistics(data, args.output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
