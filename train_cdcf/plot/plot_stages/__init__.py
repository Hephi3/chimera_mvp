"""
Cross-validation plotting package for federated learning experiments.

This package provides utilities for plotting and comparing cross-validation results
from federated learning experiments.
"""

from .plot_utils import *
from .statistics import compute_fold_statistics
from .plotting import plot_metric, create_legend_subplot
from .crossval_single import plot_cross_validation
from .crossval_comparison import plot_cross_validation_comparison

__all__ = [
    # Main functions
    'plot_cross_validation',
    'plot_cross_validation_comparison',
    
    # Statistics
    'compute_fold_statistics',
    
    # Plotting
    'plot_metric',
    'create_legend_subplot',
    
    # Utilities
    'smooth_client_data',
    'tensorboard_to_datadict_federated',
    'get_available_folds',
    'create_figure',
    'filter_metrics',
    'calculate_global_parameters',
    'hide_empty_subplots',
    'save_plot',
    'create_mode_description',
    'create_filename_suffix',
    
    # Constants
    'ROOT_RESULTS',
    'FEDERATED_METRICS',
    'LINE_STYLES',
    'SUBMODELS',
    'LINE_ALPHA',
    'STD_ALPHA',
]