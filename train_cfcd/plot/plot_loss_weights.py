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
    
    The file format is:
    slide_id: Parameter containing:
    tensor(value1, device='cuda:0', requires_grad=True), Parameter containing:
    tensor(value2, device='cuda:0', requires_grad=True), Parameter containing:
    tensor(value3, device='cuda:0', requires_grad=True)
    
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
    
    # Each entry spans 4 lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is the start of an entry (contains slide_id and "Parameter containing:")
        if ':' in line and 'Parameter containing:' in line:
            # Combine the current line and next 3 lines to get all three tensor values
            entry_text = line
            for j in range(1, 4):
                if i + j < len(lines):
                    entry_text += ' ' + lines[i + j].strip()
            
            # Extract all tensor values from the combined text
            # Pattern matches: tensor(0.0002, ...) or tensor(1.0000e-04, ...)
            tensor_pattern = r'tensor\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            tensor_values = re.findall(tensor_pattern, entry_text)
            
            if len(tensor_values) >= 3:
                try:
                    mm_val = float(tensor_values[0])
                    clam_val = float(tensor_values[1])
                    cd_val = float(tensor_values[2])
                    
                    # Apply exp transformation as done in the code: w = torch.exp(weight)
                    weights['mm'].append(np.exp(mm_val))
                    weights['clam'].append(np.exp(clam_val))
                    weights['cd'].append(np.exp(cd_val))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse entry at line {i}: {e}")
            
            # Move to the next entry (skip 4 lines)
            i += 4
        else:
            i += 1
    
    return weights


def plot_weight_curves(weights, output_path=None, title="Loss Weight Evolution"):
    """
    Plot the weight curves for MM, CLAM, and CD submodels.
    
    Args:
        weights: Dictionary with 'mm', 'clam', 'cd' keys containing weight lists
        output_path: Path to save the plot (optional)
        title: Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    # Subplot 1: All weights together
    ax1 = axes[0, 0]
    iterations = range(len(weights['mm']))
    ax1.plot(iterations, weights['mm'], label='MM', linewidth=2, alpha=0.8)
    ax1.plot(iterations, weights['clam'], label='CLAM', linewidth=2, alpha=0.8)
    ax1.plot(iterations, weights['cd'], label='CD', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Training Iteration (Slide)')
    ax1.set_ylabel('Weight (exp(parameter))')
    ax1.set_title('All Loss Weights')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: MM weight only
    ax2 = axes[0, 1]
    ax2.plot(iterations, weights['mm'], label='MM', color='C0', linewidth=2)
    ax2.set_xlabel('Training Iteration (Slide)')
    ax2.set_ylabel('Weight (exp(parameter))')
    ax2.set_title('MM Loss Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: CLAM weight only
    ax3 = axes[1, 0]
    ax3.plot(iterations, weights['clam'], label='CLAM', color='C1', linewidth=2)
    ax3.set_xlabel('Training Iteration (Slide)')
    ax3.set_ylabel('Weight (exp(parameter))')
    ax3.set_title('CLAM Loss Weight')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: CD weight only
    ax4 = axes[1, 1]
    ax4.plot(iterations, weights['cd'], label='CD', color='C2', linewidth=2)
    ax4.set_xlabel('Training Iteration (Slide)')
    ax4.set_ylabel('Weight (exp(parameter))')
    ax4.set_title('CD Loss Weight')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
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
                       default='../results/del_sp1_s3/Fold0/loss_weights.txt',
                       help='Path to loss_weights.txt file')
    parser.add_argument('--output', '-o', type=str,
                       default='loss_weights_curves.png',
                       help='Output path for the plot')
    parser.add_argument('--title', '-t', type=str,
                       default='Loss Weight Evolution',
                       help='Title for the plot')
    
    args = parser.parse_args()
    
    # Parse the file
    print(f"Reading loss weights from: {args.input}")
    weights = parse_loss_weights_file(args.input)
    
    # Print statistics
    print_statistics(weights)
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_weight_curves(weights, output_path=args.output, title=args.title)


if __name__ == '__main__':
    main()
