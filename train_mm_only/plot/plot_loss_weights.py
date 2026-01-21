#!/usr/bin/env python3
"""
Plot loss weight curves for MM, CLAM, and CD submodels from loss_weights.txt
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def parse_loss_weights_file(file_path):
    """
    Parse the loss_weights.txt file and extract weight values for each submodel.
    
    The file format is CSV:
    value1, value2, value3
    
    Returns:
        dict: Dictionary with keys 'mm', 'clam', 'cd' containing lists of weight values
    """
    weights = {
        'mm': [],
        'clam': [],
        'cd': []
    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Split by comma
        parts = line.split(',')
        
        if len(parts) >= 3:
            try:
                mm_val = float(parts[0].strip())
                clam_val = float(parts[1].strip())
                cd_val = float(parts[2].strip())
                
                weights['mm'].append(mm_val)
                weights['clam'].append(clam_val)
                weights['cd'].append(cd_val)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {i+1}: {line} - {e}")
    
    return weights


def plot_weight_curves(weights, output_path=None, title="Loss Weight Evolution", start_index=25):
    """
    Plot the weight curves for MM, CLAM, and CD submodels.
    
    Args:
        weights: Dictionary with 'mm', 'clam', 'cd' keys containing weight lists
        output_path: Path to save the plot (optional)
        title: Title for the plot
        start_index: Starting index to plot from (default: 25, to skip initial constant values)
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Slice the data to start from start_index
    iterations = range(start_index, len(weights['mm']))
    
    ax.plot(iterations, weights['cd'][start_index:], label='Clinical Data', linewidth=2, alpha=0.8)
    ax.plot(iterations, weights['clam'][start_index:], label='Histopathological Data', linewidth=2, alpha=0.8,)
    ax.plot(iterations, weights['mm'][start_index:], label='Multimodal Fusion', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Iteration (Epoch)', fontsize=12)
    ax.set_ylabel('Loss Weight', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # plt.show()
    
    return fig


def print_statistics(weights):
    """Print statistics about the weights."""
    print("\n=== Loss Weight Statistics ===")
    for key in ['mm', 'clam', 'cd']:
        w = np.array(weights[key])
        print(f"\n{key.upper()}:")
        print(f"  Count: {len(w)}")
        if len(w) > 0:
            print(f"  Mean: {w.mean():.6f}")
            print(f"  Std: {w.std():.6f}")
            print(f"  Min: {w.min():.6f}")
            print(f"  Max: {w.max():.6f}")
            print(f"  Final value: {w[-1]:.6f}")
        else:
            print("  No data found!")


def main():
    parser = argparse.ArgumentParser(description='Plot loss weight curves from loss_weights.txt')
    parser.add_argument('--input', '-i', type=str, 
                       default='../results/del3_s1/loss_weights_epochs.txt',
                       help='Path to loss_weights.txt file')
    parser.add_argument('--output', '-o', type=str,
                       default='loss_weights_curves.png',
                       help='Output path for the plot')
    parser.add_argument('--title', '-t', type=str,
                       default='Loss Weight Evolution',
                       help='Title for the plot')
    parser.add_argument('--start', '-s', type=int,
                       default=25,
                       help='Starting index to plot from (default: 25)')
    
    args = parser.parse_args()
    
    # Parse the file
    print(f"Reading loss weights from: {args.input}")
    weights = parse_loss_weights_file(args.input)
    
    # Print statistics
    print_statistics(weights)
    
    # Create plot
    print(f"\nGenerating plot starting from index {args.start}...")
    plot_weight_curves(weights, output_path=args.output, title=args.title, start_index=args.start)


if __name__ == '__main__':
    main()
