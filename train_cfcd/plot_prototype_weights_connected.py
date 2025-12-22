"""
Script to plot prototype weights with connected quartiles and whiskers for thesis.
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


def plot_weights_connected(weights, num_rounds=40, show_outliers=False, output_path=None):
    """Create a connected plot showing quartiles and whiskers as shaded areas."""
    
    # Calculate samples per round
    total_samples = len(weights)
    samples_per_round = total_samples // num_rounds
    
    if total_samples % num_rounds != 0:
        print(f"Warning: {total_samples} samples doesn't divide evenly by {num_rounds} rounds")
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
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    
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
    
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Loss Weight per Samples', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot prototype weights with connected distribution')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to prototype_weights.txt file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot (e.g., prototype_weights.png)')
    parser.add_argument('--rounds', type=int, default=40,
                       help='Number of training rounds (default: 40)')
    parser.add_argument('--show_outliers', action='store_true',
                       help='Show outliers as scatter points')
    
    args = parser.parse_args()
    
    # Parse data
    print(f"Reading data from: {args.input}")
    slide_ids, weights = parse_weights_file(args.input)
    
    print(f"Loaded {len(weights)} prototype weights")
    print(f"Overall mean weight: {np.mean(weights):.4f}")
    print(f"Overall std weight: {np.std(weights):.4f}")
    
    # Set output path
    if args.output is None:
        args.output = Path(args.input).parent / 'prototype_weights_connected2.png'
    
    # Create plot
    plot_weights_connected(weights, args.rounds, args.show_outliers, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()


# /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_ML_01_str5_sp2_s2/Fold0/prototype_weights.txt

# /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_ML_02_2str_sp1_s1/Fold0/prototype_weights.txt