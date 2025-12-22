import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_aggregation_weights(filepath):
    """Plot the evolution of aggregation weights over rounds."""
    # Read the weights file
    weights = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                w = [float(x) for x in line.split(',')]
                weights.append(w)
    
    weights = np.array(weights)
    num_rounds = len(weights)
    num_clients = weights.shape[1]
    
    # Create the plot
    plt.figure(figsize=(5, 3))
    
    for i in range(num_clients):
        plt.plot(range(1, num_rounds + 1), weights[:, i], 
                label=f'Client {i}', linewidth=2)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Aggregation Weight', fontsize=12)
    plt.legend(fontsize=10)
    temperature = filepath.split('_')[3][1:-1]
    plt.title(f'Temperature T=0.{temperature}')
    plt.grid(True, alpha=0.3)
    # plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    # Save the plot
    output_path = f"aggregation_weights_plot_{filepath.split('/')[1]}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        raise ValueError("Please provide the path to aggregation_weights.txt as a command line argument.")
    plot_aggregation_weights(filepath)


# python plot_aggregation_weights.py results/73MG_02_02t_sp2_s3/Fold1/aggregation_weights.txt

# /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_MG_01_002t_sp1_s1/Fold0/aggregation_weights.txt
# /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_MG_05_01t_sp1_s1/Fold0/aggregation_weights.txt
# /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/73_MG_03_05t_sp1_s1/Fold0/aggregation_weights.txt
# /gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/MG_08_sp1_s3/Fold0/aggregation_weights.txt