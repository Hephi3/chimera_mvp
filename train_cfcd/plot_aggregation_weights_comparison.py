import numpy as np
import matplotlib.pyplot as plt
import os

def load_weights(filepath):
    """Load aggregation weights from file."""
    weights = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                w = [float(x) for x in line.split(',')]
                weights.append(w)
    return np.array(weights)

def plot_temperature_comparison():
    """Plot aggregation weights comparison for different temperature values."""
    
    # Define the experiments with their paths and temperature values
    experiments = [
        {
            'path': '/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_MG_05_01t_sp1_s1/Fold4/aggregation_weights.txt',
            'temp': 0.1,
            'label': 'T=0.1',
            'color': '#3498db'  # blue
        },
        {
            'path': '/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_MG_03_05t_sp1_s1/Fold2/aggregation_weights.txt',
            'temp': 0.5,
            'label': 'T=0.5',
            'color': '#2ecc71'  # green
        },
        {
            'path': '/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/MG_08_sp1_s3/Fold1/aggregation_weights.txt',
            'temp': 1.0,
            'label': 'T=1.0',
            'color': '#e74c3c'  # red
        }
    ]
    
    # Create single figure
    plt.figure(figsize=(10, 6))
    
    # Different line styles for each client
    client_styles = [
        {'linestyle': '-', 'marker': 'o'},      # Client 0: solid
        {'linestyle': '--', 'marker': 's'},     # Client 1: dashed
        {'linestyle': '-.', 'marker': '^'},     # Client 2: dash-dot
    ]
    
    # Plot data
    for idx, exp in enumerate(experiments):
        if not os.path.exists(exp['path']):
            print(f"Warning: File not found: {exp['path']}")
            continue
        
        weights = load_weights(exp['path'])
        num_rounds = len(weights)
        num_clients = weights.shape[1]
        
        # Plot all clients for this temperature
        for client_idx in range(num_clients):
            style = client_styles[client_idx % len(client_styles)]
            plt.plot(range(1, num_rounds + 1), weights[:, client_idx], 
                   color=exp['color'],
                   linestyle=style['linestyle'],
                   marker=style['marker'],
                   markersize=4,
                   linewidth=2,
                   markevery=4,
                   alpha=0.85)
    
    # Create custom legend with 6 entries
    from matplotlib.lines import Line2D
    
    # Legend entries for temperatures (colored lines)
    temp_handles = [
        Line2D([0], [0], color=exp['color'], linewidth=2, label=exp['label'])
        for exp in experiments
    ]
    
    # Legend entries for clients (black lines with different styles)
    client_handles = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Client 1'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Client 2'),
        Line2D([0], [0], color='black', linestyle='-.', linewidth=2, label='Client 3'),
    ]
    
    # Combine handles
    all_handles = temp_handles + client_handles
    
    plt.xlabel('Round', fontsize=13)
    plt.ylabel('Aggregation Weight', fontsize=13)
    plt.legend(handles=all_handles, fontsize=10, loc='best', framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # Save the plot
    output_path = '/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/temperature_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_temperature_comparison()
