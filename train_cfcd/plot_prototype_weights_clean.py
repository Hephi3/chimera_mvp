"""
Script to plot prototype weights in a clean format for thesis.
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
    
    return slide_ids, np.array(weights)


def plot_weights_clean(weights, num_rounds=40, show_outliers=False, output_path=None):
    """Create a clean plot of prototype weights for thesis."""
    
    # Calculate samples per round
    total_samples = len(weights)
    samples_per_round = total_samples // num_rounds
    
    if total_samples % num_rounds != 0:
        print(f"Warning: {total_samples} samples doesn't divide evenly by {num_rounds} rounds")
        print(f"Using {samples_per_round} samples per round, ignoring last {total_samples % num_rounds} samples")
    
    # Reshape weights into rounds
    weights_per_round = weights[:samples_per_round * num_rounds].reshape(num_rounds, samples_per_round)
    
    # Calculate mean and std for each round
    mean_weights = np.mean(weights_per_round, axis=1)
    std_weights = np.std(weights_per_round, axis=1)
    
    if show_outliers:
        # Create boxplot to show distributions
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Create boxplot
        bp = ax.boxplot([weights_per_round[i, :] for i in range(num_rounds)],
                        positions=np.arange(1, num_rounds + 1),
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanline=False,
                        boxprops=dict(facecolor='#1f77b4', alpha=0.5),
                        medianprops=dict(color='#ff7f0e', linewidth=1.5),
                        meanprops=dict(marker='D', markerfacecolor='#d62728', markeredgecolor='#d62728', markersize=4),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))
        
        # Overlay mean line
        rounds = np.arange(1, num_rounds + 1)
        ax.plot(rounds, mean_weights, linewidth=1.5, color='#d62728', label='Mean', zorder=10, linestyle="-")
        
        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Prototype Weight', fontsize=12)
        ax.set_title('Prototype Weight Distribution per Training Round', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.legend(fontsize=10)
        
    else:
        # Original plot with mean and std
        fig, ax = plt.subplots(figsize=(10, 4))
        
        rounds = np.arange(1, num_rounds + 1)
        
        # Plot mean with shaded std area
        ax.plot(rounds, mean_weights, linewidth=1.5, color='#1f77b4', label='Mean')
        ax.fill_between(rounds, mean_weights - std_weights, mean_weights + std_weights, 
                         alpha=0.3, color='#1f77b4', label='Std Dev')
        
        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Prototype Weight', fontsize=12)
        ax.set_title('Mean Prototype Weight Over Training Rounds', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot prototype weights in clean format')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to prototype_weights.txt file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot (e.g., prototype_weights.png)')
    parser.add_argument('--rounds', type=int, default=40,
                       help='Number of training rounds (default: 40)')
    parser.add_argument('--show_outliers', action='store_true',
                       help='Show min/max values (outliers) for each round')
    
    args = parser.parse_args()
    
    # Parse data
    print(f"Reading data from: {args.input}")
    slide_ids, weights = parse_weights_file(args.input)
    
    print(f"Loaded {len(weights)} prototype weights")
    print(f"Overall mean weight: {np.mean(weights):.4f}")
    print(f"Overall std weight: {np.std(weights):.4f}")
    
    # Set output path
    if args.output is None:
        args.output = Path(args.input).parent / 'prototype_weights_clean.png'
    
    # Create plot
    plot_weights_clean(weights, args.rounds, args.show_outliers, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
