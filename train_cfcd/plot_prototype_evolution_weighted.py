"""
Modified version that colors samples by their average weights.
Since slide IDs aren't saved in debug files, we'll infer the mapping
based on the order slides appear in the weights file.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def load_weights_from_file(weights_file):
    """Load weights and return average weight per slide ID."""
    weights_by_slide = defaultdict(list)
    slide_order = []
    
    try:
        with open(weights_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                
                slide_id = parts[0]
                try:
                    weight = float(parts[1])
                    if slide_id not in weights_by_slide:
                        slide_order.append(slide_id)
                    weights_by_slide[slide_id].append(weight)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Warning: Weights file not found: {weights_file}")
        return {}, []
    
    # Calculate average weights
    avg_weights = {sid: np.mean(weights) for sid, weights in weights_by_slide.items()}
    
    return avg_weights, slide_order


def plot_prototype_evolution_weighted(samples_per_round, labels_per_round, avg_weights, slide_order,
                                      num_clients=3, client_colors=None, show_samples=True, sample_alpha=0.3):
    """
    Plot prototype evolution with samples colored by their average weight.
    
    Args:
        samples_per_round: List of lists [round][client] -> features
        labels_per_round: List of lists [round][client] -> labels  
        avg_weights: Dictionary mapping slide_id to average weight
        slide_order: List of slide IDs in order they appear
        num_clients: Number of clients
        client_colors: Colors for client prototypes
        show_samples: Whether to show individual samples
        sample_alpha: Alpha for individual samples
    """
    if client_colors is None:
        client_colors = ['blue', 'green', 'orange']
    
    num_rounds = len(samples_per_round)
    
    # Collect all data for PCA
    all_data = []
    for round_samples in samples_per_round:
        for client_samples in round_samples:
            all_data.append(client_samples)
    all_data = np.vstack(all_data)
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Setup colormap for weights
    if avg_weights:
        vmin, vmax = min(avg_weights.values()), max(avg_weights.values())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('RdYlGn')  # Red (low) to Green (high)
    else:
        cmap = None
        norm = None
    
    # Alpha and size values for rounds
    alphas = np.linspace(0.3, 1.0, num_rounds)
    sizes = np.linspace(3, 12, num_rounds)
    
    # Track client prototypes
    client_proto_positions = [[] for _ in range(num_clients)]
    
    # Build slide ID to weight mapping based on order
    slide_idx_counter = 0
    
    # Plot samples
    if show_samples and cmap is not None:
        for round_idx, (round_samples, round_labels) in enumerate(zip(samples_per_round, labels_per_round)):
            round_alpha = alphas[round_idx] * sample_alpha
            round_size = sizes[round_idx]
            
            for client_idx in range(num_clients):
                samples_2d = pca.transform(round_samples[client_idx])
                
                # For each sample, assign a weight based on order
                for sample_2d in samples_2d:
                    if slide_idx_counter < len(slide_order):
                        slide_id = slide_order[slide_idx_counter]
                        weight = avg_weights.get(slide_id, 0.5)
                        color = cmap(norm(weight))
                        slide_idx_counter += 1
                    else:
                        # Cycle through if we run out
                        slide_id = slide_order[slide_idx_counter % len(slide_order)]
                        weight = avg_weights.get(slide_id, 0.5)
                        color = cmap(norm(weight))
                        slide_idx_counter += 1
                    
                    ax.scatter(sample_2d[0], sample_2d[1], 
                             c=[color], s=round_size, alpha=round_alpha,
                             edgecolors='none', zorder=0)
    elif show_samples:
        # Fallback: use client colors
        for round_idx, round_samples in enumerate(samples_per_round):
            round_alpha = alphas[round_idx] * sample_alpha
            round_size = sizes[round_idx]
            for client_idx in range(num_clients):
                samples_2d = pca.transform(round_samples[client_idx])
                ax.scatter(samples_2d[:, 0], samples_2d[:, 1],
                          c=client_colors[client_idx], s=round_size, alpha=round_alpha,
                          edgecolors='none', zorder=0)
    
    # Plot prototypes
    for round_idx, round_samples in enumerate(samples_per_round):
        alpha = alphas[round_idx]
        
        prototypes = [Prototype.from_data(round_samples[i]) for i in range(num_clients)]
        global_proto = Prototype.from_prototypes(prototypes)
        
        # Global prototype
        global_proto_2d = pca.transform(global_proto.mean.reshape(1, -1))
        global_label = 'Global Prototype' if round_idx == num_rounds - 1 else None
        ax.scatter(global_proto_2d[:, 0], global_proto_2d[:, 1], c='red',
                  marker='X', s=120, alpha=alpha, edgecolors='black', 
                  linewidths=1, zorder=2, label=global_label)
        
        # Global variance ellipse
        global_cov = np.diag(global_proto.var)
        global_cov_2d = pca.components_ @ global_cov @ pca.components_.T
        vals, vecs = np.linalg.eigh(global_cov_2d)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(vals)
        ellipse = Ellipse(xy=global_proto_2d[0], width=width, height=height, angle=theta,
                         edgecolor='red', fc='None', lw=1.5, alpha=alpha*0.4, zorder=1)
        ax.add_patch(ellipse)
        
        # Client prototypes
        for client_idx in range(num_clients):
            proto_2d = pca.transform(prototypes[client_idx].mean.reshape(1, -1))
            client_proto_positions[client_idx].append(proto_2d[0])
            
            label = f'Client {client_idx + 1}' if round_idx == num_rounds - 1 else None
            ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=client_colors[client_idx],
                      marker='o', s=120, alpha=alpha, label=label,
                      edgecolors='black', linewidths=1.5, zorder=2)
            
            # Variance ellipse
            cov = np.diag(prototypes[client_idx].var)
            cov_2d = pca.components_ @ cov @ pca.components_.T
            vals, vecs = np.linalg.eigh(cov_2d)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 2 * np.sqrt(vals)
            ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta,
                            edgecolor=client_colors[client_idx], fc='None', 
                            lw=2, alpha=alpha*0.6, zorder=1)
            ax.add_patch(ellipse)
    
    # Trajectory lines
    for client_idx in range(num_clients):
        if len(client_proto_positions[client_idx]) > 1:
            positions = np.array(client_proto_positions[client_idx])
            ax.plot(positions[:, 0], positions[:, 1],
                   color=client_colors[client_idx], linestyle='--', 
                   alpha=0.6, linewidth=2, zorder=1)
            
            # Arrows
            for i in range(len(positions) - 1):
                dx = positions[i+1, 0] - positions[i, 0]
                dy = positions[i+1, 1] - positions[i, 1]
                ax.arrow(positions[i, 0], positions[i, 1],
                        dx*0.7, dy*0.7, head_width=0.15, head_length=0.12,
                        fc=client_colors[client_idx], ec=client_colors[client_idx],
                        alpha=0.5, zorder=1)
    
    ax.set_xlabel("PC 1", fontsize=18)
    ax.set_ylabel("PC 2", fontsize=18)
    ax.legend(loc='upper right', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    if cmap is not None and norm is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Average Prototype Weight', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    return fig, ax


def load_features_from_rounds(base_dir, num_clients=3, rounds=None):
    """Load features and labels from debug files."""
    if rounds is None:
        rounds = list(range(1, 11))
    
    samples_per_round = []
    labels_per_round = []
    
    for round_num in rounds:
        round_samples = []
        round_labels = []
        
        for client_id in range(num_clients):
            filepath = f"{base_dir}/debug_features_client_{client_id}_round_{round_num}.npz"
            try:
                data = np.load(filepath)
                features = data["features"]
                if features.ndim == 3:
                    features = features.squeeze(1)
                round_samples.append(features)
                
                if "labels" in data:
                    round_labels.append(data["labels"])
                else:
                    round_labels.append(np.zeros(len(features)))
                    
            except FileNotFoundError:
                print(f"Warning: File not found: {filepath}")
                if len(round_samples) > 0:
                    round_samples.append(round_samples[-1])
                    round_labels.append(round_labels[-1])
                else:
                    break
        
        if len(round_samples) == num_clients:
            samples_per_round.append(round_samples)
            labels_per_round.append(round_labels)
    
    return samples_per_round, labels_per_round


if __name__ == "__main__":
    base_dir = "results/73_ML_02_str1_sp1_s3/Fold0"
    weights_file = f"{base_dir}/prototype_weights.txt"
    rounds_to_plot = list(range(1, 61))  # Adjust as needed
    
    print(f"Loading data from: {base_dir}")
    avg_weights, slide_order = load_weights_from_file(weights_file)
    samples_per_round, labels_per_round = load_features_from_rounds(base_dir, num_clients=3, rounds=rounds_to_plot)
    
    if len(samples_per_round) > 0 and avg_weights:
        print(f"Loaded {len(samples_per_round)} rounds")
        print(f"Loaded {len(avg_weights)} unique slides with weights")
        print(f"Weight range: {min(avg_weights.values()):.3f} - {max(avg_weights.values()):.3f}")
        
        fig, ax = plot_prototype_evolution_weighted(
            samples_per_round, labels_per_round, avg_weights, slide_order,
            num_clients=3, show_samples=True, sample_alpha=0.4
        )
        
        output_file = f"{base_dir}/prototype_evolution_weighted.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as: {output_file}")
        plt.close()
    else:
        print("Failed to load data or weights. Check file paths.")
