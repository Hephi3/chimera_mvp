#!/usr/bin/env python3
"""
Cross-Validation Comparison Plotting Script

This script compares cross-validation results from multiple federated learning experiments.
It provides a simplified interface for comparing performance across different experimental setups.

Usage:
    python plot_comparison_crossval.py --names EXP1 EXP2 [EXP3 ...] [options]

Examples:
    # Basic comparison of two experiments
    python plot_comparison_crossval.py --names CF_s1 CF_s2 --submodel MM --metric_filter all

    # Compare with individual client performance
    python plot_comparison_crossval.py --names CF_s1 CF_s2 --submodel MM --metric_filter all --show_individual_clients

    # Compare full training curves with smoothing
    python plot_comparison_crossval.py --names CF_s1 CF_s2 CF_s3 --submodel MM --metric_filter all --show_full_training --smooth_window 5

    # Compare only test metrics
    python plot_comparison_crossval.py --names CF_s1 CF_s2 --submodel MM --metric_filter test
"""

import argparse
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossval_comparison import plot_cross_validation_comparison


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Validation Comparison Plotting for Multiple Federated Learning Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison of two experiments
  %(prog)s --names CF_s1 CF_s2 --submodel MM --metric_filter all

  # Compare with individual client performance
  %(prog)s --names CF_s1 CF_s2 --submodel MM --metric_filter all --show_individual_clients

  # Compare full training curves with smoothing
  %(prog)s --names CF_s1 CF_s2 CF_s3 --submodel MM --metric_filter all --show_full_training --smooth_window 5

  # Compare only test metrics for specific folds
  %(prog)s --names CF_s1 CF_s2 --submodel MM --metric_filter test --folds 0 1 2

  # Compare without standard deviation bands  
  %(prog)s --names CF_s1 CF_s2 --submodel MM --metric_filter all --no_std
        """
    )
    
    # Required arguments
    parser.add_argument('--names', nargs='+', required=True,
                       help='Names of experiments to compare (each should contain fold directories)')
    
    # Optional arguments
    parser.add_argument('--submodel', type=str, default='MM', choices=['MM', 'CLAM', 'CD'], 
                       help='Which submodel to plot (default: MM)')
    parser.add_argument('--metric_filter', type=str, default='all', 
                       choices=['train', 'val', 'test', 'all'],
                       help='Which metrics to show (default: all)')
    parser.add_argument('--show_individual_clients', action='store_true', 
                       help='Show individual client statistics instead of aggregated')
    parser.add_argument('--show_full_training', action='store_true',
                       help='Show complete training progress within rounds instead of just final values')
    parser.add_argument('--smooth_window', type=int, default=0,
                       help='Apply moving average smoothing with this window size (0 = no smoothing)')
    parser.add_argument('--show_std', action='store_true', default=True,
                       help='Show standard deviation as bands/error bars (default: True)')
    parser.add_argument('--no_std', action='store_true',
                       help='Do not show standard deviation (overrides --show_std)')
    parser.add_argument('--folds', nargs='+', type=int,
                       help='Specific fold numbers to include (default: use all available)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.names) < 2:
        print("Error: At least two experiment names are required for comparison")
        sys.exit(1)
    
    # Handle std flag logic
    show_std = args.show_std and not args.no_std
    
    try:
        plot_path = plot_cross_validation_comparison(
            experiment_names=args.names,
            submodel=args.submodel,
            metric_filter=args.metric_filter,
            show_individual_clients=args.show_individual_clients,
            show_full_training=args.show_full_training,
            smooth_window=args.smooth_window,
            show_std=show_std,
            folds=args.folds
        )
        
        print(f"Successfully created cross-validation comparison plot: {plot_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()