#!/usr/bin/env python3

import argparse
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossval_single import plot_cross_validation


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Validation Plotting for Single Federated Learning Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required arguments
    parser.add_argument('--name', type=str, required=True,
                       help='Name of the experiment (should contain fold directories)')
    
    # Optional arguments
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD'], 
                       help='Which submodel to plot (default: MM)')
    parser.add_argument('--metric_filter', type=str, default='all', 
                       choices=['train', 'val', 'test', 'all'],
                       help='Which metrics to show (default: all)')
    parser.add_argument('--show_individual_clients', action='store_true', default=True,
                       help='Show individual client statistics instead of aggregated')
    parser.add_argument('--show_full_training', action='store_true', default=True,
                       help='Show complete training progress within rounds instead of just final values')
    parser.add_argument('--smooth_window', type=int, default=5,
                       help='Apply moving average smoothing with this window size (0 = no smoothing)')
    parser.add_argument('--show_std', action='store_true', default=True,
                       help='Show standard deviation as bands/error bars (default: True)')
    parser.add_argument('--no_std', action='store_true',
                       help='Do not show standard deviation (overrides --show_std)')
    parser.add_argument('--folds', nargs='+', type=int,
                       help='Specific fold numbers to include (default: use all available)')
    
    args = parser.parse_args()
    
    print(args)
    
    # Handle std flag logic
    show_std = args.show_std and not args.no_std
    
    try:
        plot_path = plot_cross_validation(
            experiment_name=args.name,
            submodel=args.submodel,
            metric_filter=args.metric_filter,
            show_individual_clients=args.show_individual_clients,
            show_full_training=args.show_full_training,
            smooth_window=args.smooth_window,
            show_std=show_std,
            folds=args.folds
        )
        
        print(f"Successfully created cross-validation plot: {plot_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
# python plot_single_crossval.py --name CF_s1

# Plot only specific folds
# python plot_single_crossval.py --name CF_s1 --folds 0 1 2