"""
Script to compare prototype weights from two experiments side by side.
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


def plot_single_experiment(ax, weights, num_rounds, show_outliers, title):
    """Plot a single experiment on the given axis."""
    
    # Calculate samples per round
    total_samples = len(weights)
    samples_per_round = total_samples // num_rounds
    
    if total_samples % num_rounds != 0:
        print(f"Warning ({title}): {total_samples} samples doesn't divide evenly by {num_rounds} rounds")
        print(f"Using {samples_per_round} samples per round, ignoring last {total_samples % num_rounds} samples")
    
    # Reshape weights into rounds
    weights_per_round = weights[:samples_per_round * num_rounds].reshape(num_rounds, samples_per_round)
    
    # Calculate statistics for each round
    rounds = np.arange(1, num_rounds + 1)
    means = np.mean(weights_per_round, axis=1)
    medians = np.median(weights_per_round, axis=1)
    q1 = np.percentile(weights_per_round, 25, axis=1)
    q3 = np.percentile(weights_per_round, 75, axis=1)
    
    # Calculate whiskers (1.5 * IQR)
    iqr = q3 - q1
    lower_whisker = np.maximum(q1 - 1.5 * iqr, np.min(weights_per_round, axis=1))
    upper_whisker = np.minimum(q3 + 1.5 * iqr, np.max(weights_per_round, axis=1))
    
    # Plot shaded areas from outer to inner
    # Whiskers range (lightest)
    ax.fill_between(rounds, lower_whisker, upper_whisker, 
                     alpha=0.2, color='#1f77b4', label='Whisker Range')
    
    # Quartile range (IQR) - darker
    ax.fill_between(rounds, q1, q3, 
                     alpha=0.4, color='#1f77b4', label='IQR (Q1-Q3)')
    
    # Plot outliers if requested
    outlier_plotted = False
    if show_outliers:
        for i in range(num_rounds):
            round_weights = weights_per_round[i, :]
            # Identify outliers (values outside whisker range)
            outliers = round_weights[(round_weights < lower_whisker[i]) | (round_weights > upper_whisker[i])]
            if len(outliers) > 0:
                if not outlier_plotted:
                    ax.scatter([rounds[i]] * len(outliers), outliers, 
                              color='red', s=8, alpha=0.6, zorder=15, label='Outliers')
                    outlier_plotted = True
                else:
                    ax.scatter([rounds[i]] * len(outliers), outliers, 
                              color='red', s=8, alpha=0.6, zorder=15)
    
    # Plot lines
    ax.plot(rounds, medians, linewidth=2, color='#ff7f0e', label='Median', zorder=10)
    ax.plot(rounds, means, linewidth=2, color='#d62728', label='Mean', linestyle='--', zorder=10)
    
    ax.set_xlabel('Training Round', fontsize=11)
    ax.set_ylabel('Loss Weight per Sample', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.legend(fontsize=9, loc='best')


def plot_comparison(weights1, weights2, num_rounds, show_outliers, title1, title2, output_path=None):
    """Create comparison plot with two experiments side by side."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_single_experiment(ax1, weights1, num_rounds, show_outliers, title1)
    plot_single_experiment(ax2, weights2, num_rounds, show_outliers, title2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare prototype weights from two experiments')
    parser.add_argument('--input1', type=str, required=True,
                       help='Path to first prototype_weights.txt file')
    parser.add_argument('--input2', type=str, required=True,
                       help='Path to second prototype_weights.txt file')
    parser.add_argument('--title1', type=str, default='Experiment 1',
                       help='Title for first experiment')
    parser.add_argument('--title2', type=str, default='Experiment 2',
                       help='Title for second experiment')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot (e.g., comparison.png)')
    parser.add_argument('--rounds', type=int, default=40,
                       help='Number of training rounds (default: 40)')
    parser.add_argument('--show_outliers', action='store_true',
                       help='Show outliers as scatter points')
    
    args = parser.parse_args()
    
    # Parse data
    print(f"Reading data from: {args.input1}")
    _, weights1 = parse_weights_file(args.input1)
    print(f"  Loaded {len(weights1)} weights, mean: {np.mean(weights1):.4f}")
    
    print(f"Reading data from: {args.input2}")
    _, weights2 = parse_weights_file(args.input2)
    print(f"  Loaded {len(weights2)} weights, mean: {np.mean(weights2):.4f}")
    
    # Set output path
    if args.output is None:
        args.output = 'prototype_weights_comparison.png'
    
    # Create comparison plot
    plot_comparison(weights1, weights2, args.rounds, args.show_outliers, 
                   args.title1, args.title2, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
