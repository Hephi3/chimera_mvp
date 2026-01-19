import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


def plot_prototype_evolution(samples_per_round, num_clients=3, client_colors=None, show_samples=True, sample_alpha=0.05):
    """
    Plot the evolution of prototypes over multiple rounds.
    
    Args:
        samples_per_round: List of lists, where samples_per_round[round][client] is the feature array for that client at that round
        num_clients: Number of clients (default 3)
        client_colors: List of colors for each client (default ['blue', 'green', 'orange'])
        show_samples: Whether to show individual data samples (default True)
        sample_alpha: Alpha value for individual samples (default 0.05)
    """
    if client_colors is None:
        client_colors = ['blue', 'green', 'orange']
    
    num_rounds = len(samples_per_round)
    
    # Collect all data for PCA fitting
    all_data = []
    for round_samples in samples_per_round:
        for client_samples in round_samples:
            all_data.append(client_samples)
    all_data = np.vstack(all_data)
    
    # Fit PCA on all data
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Alpha values decrease for older rounds (most recent = 1.0, oldest = 0.2)
    alphas = np.linspace(0.2, 1.0, num_rounds)
    # Size values increase for newer rounds (oldest = 2, newest = 10)
    sizes = np.linspace(2, 10, num_rounds)
    
    # Track client prototypes for trajectories
    client_proto_positions = [[] for _ in range(num_clients)]
    
    # First pass: Plot individual samples in background (if enabled)
    if show_samples:
        for round_idx, round_samples in enumerate(samples_per_round):
            round_alpha = alphas[round_idx] * sample_alpha
            round_size = sizes[round_idx]
            for client_idx in range(num_clients):
                # Transform samples to 2D
                samples_2d = pca.transform(round_samples[client_idx])
                # Plot samples as dots with increasing size for newer rounds
                ax.scatter(samples_2d[:, 0], samples_2d[:, 1], 
                          c=client_colors[client_idx], s=round_size, alpha=round_alpha, 
                          edgecolors='none', zorder=0)
    
    # Second pass: Plot prototypes and their variance
    for round_idx, round_samples in enumerate(samples_per_round):
        alpha = alphas[round_idx]
        
        # Create prototypes for this round
        prototypes = [Prototype.from_data(round_samples[i]) for i in range(num_clients)]
        global_proto = Prototype.from_prototypes(prototypes)
        
        # Transform global to 2D
        global_proto_2d = pca.transform(global_proto.mean.reshape(1, -1))
        
        # Plot global prototype for this round (smaller, less prominent)
        global_label = 'Global Prototype' if round_idx == num_rounds - 1 else None
        ax.scatter(global_proto_2d[:, 0], global_proto_2d[:, 1], c='red', 
                  marker='X', s=100, alpha=alpha, edgecolors='black', linewidths=0.5, zorder=1, label=global_label)
        
        # Plot variance ellipse for global prototype
        global_cov_128 = np.diag(global_proto.var)
        global_cov_2d = pca.components_ @ global_cov_128 @ pca.components_.T
        global_vals, global_vecs = np.linalg.eigh(global_cov_2d)
        global_order = global_vals.argsort()[::-1]
        global_vals = global_vals[global_order]
        global_vecs = global_vecs[:, global_order]
        global_theta = np.degrees(np.arctan2(*global_vecs[:, 0][::-1]))
        global_width, global_height = 2 * 2 * np.sqrt(global_vals)  # 2 stddev
        global_ellipse = Ellipse(xy=global_proto_2d[0], width=global_width, height=global_height, angle=global_theta,
                        edgecolor='red', fc='None', lw=1, alpha=alpha*0.3, zorder=1)
        ax.add_patch(global_ellipse)
        
        # Plot each client
        for client_idx in range(num_clients):
            proto_2d = pca.transform(prototypes[client_idx].mean.reshape(1, -1))
            client_proto_positions[client_idx].append(proto_2d[0])
            
            # Plot prototype mean
            label = f'Client {client_idx + 1}' if round_idx == num_rounds - 1 else None
            ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=client_colors[client_idx], 
                      marker='o', s=100, alpha=alpha, label=label, edgecolors='black', linewidths=1, zorder=2)
            
            # Plot variance ellipse for ALL rounds (with varying alpha)
            cov_128 = np.diag(prototypes[client_idx].var)
            cov_2d = pca.components_ @ cov_128 @ pca.components_.T
            vals, vecs = np.linalg.eigh(cov_2d)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
            ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta,
                            edgecolor=client_colors[client_idx], fc='None', lw=1.5, alpha=alpha*0.5, zorder=1)
            ax.add_patch(ellipse)
    
    # Draw trajectory lines for each client
    for client_idx in range(num_clients):
        if len(client_proto_positions[client_idx]) > 1:
            positions = np.array(client_proto_positions[client_idx])
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=client_colors[client_idx], linestyle='--', alpha=0.6, linewidth=2,) 
                #    label=f'Client {client_idx} Trajectory', zorder=1)
            
            # Add arrows to show direction
            for i in range(len(positions) - 1):
                dx = positions[i+1, 0] - positions[i, 0]
                dy = positions[i+1, 1] - positions[i, 1]
                ax.arrow(positions[i, 0], positions[i, 1], 
                        dx*0.7, dy*0.7, head_width=0.1, head_length=0.08, 
                        fc=client_colors[client_idx], ec=client_colors[client_idx], 
                        alpha=0.4, zorder=1)
    
    # ax.set_title(f"Prototype Evolution Over {num_rounds-1} Rounds", fontsize=16)
    ax.set_xlabel("PC 1", fontsize=18)
    ax.set_ylabel("PC 2", fontsize=18)
    ax.legend(loc='upper right', fontsize=18, bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    # Instead store and print in console where it has been stored
    
    plt.savefig("prototype_evolution.png")
    print("Plot saved as prototype_evolution.png")


def load_features_from_rounds(base_dir, num_clients=3, rounds=None):
    """
    Load features from multiple rounds for multiple clients.
    
    Args:
        base_dir: Base directory containing the feature files
        num_clients: Number of clients
        rounds: List of round numbers to load (e.g., [1, 2, 3, 4, 5])
    
    Returns:
        samples_per_round: List of lists [round][client] -> features
    """
    if rounds is None:
        rounds = list(range(1, 11))  # Default to rounds 1-10
    
    samples_per_round = []
    for round_num in rounds:
        round_samples = []
        for client_id in range(num_clients):
            filepath = f"{base_dir}/debug_features_client_{client_id}_round_{round_num}.npz"
            try:
                data = np.load(filepath)
                features = data["features"].squeeze(1) if data["features"].ndim == 3 else data["features"]
                round_samples.append(features)
            except FileNotFoundError:
                print(f"Warning: File not found: {filepath}")
                # Use previous round's data or skip
                if len(round_samples) > 0:
                    round_samples.append(round_samples[-1])  # Duplicate previous client
                else:
                    print(f"Skipping round {round_num}")
                    break
        if len(round_samples) == num_clients:
            samples_per_round.append(round_samples)
    
    return samples_per_round


def plot_with_synthetic_data():
    """Generate synthetic data that evolves over time and plot."""
    np.random.seed(42)
    N = 100
    D = 128
    num_rounds = 10
    num_clients = 3
    
    # Initial means for each client
    mean_c0_init = np.random.randn(D) * 1.0
    mean_c1_init = mean_c0_init + 2.0 + np.random.randn(D) * 0.5
    mean_c2_init = mean_c0_init - 1.5 + np.random.randn(D) * 0.5
    
    samples_per_round = []
    
    for round_num in range(num_rounds):
        # Gradually shift means towards a common point (simulating convergence)
        convergence_factor = round_num / num_rounds * 0.5
        target_mean = (mean_c0_init + mean_c1_init + mean_c2_init) / 3
        
        mean_c0 = mean_c0_init + convergence_factor * (target_mean - mean_c0_init) + np.random.randn(D) * 0.1
        mean_c1 = mean_c1_init + convergence_factor * (target_mean - mean_c1_init) + np.random.randn(D) * 0.1
        mean_c2 = mean_c2_init + convergence_factor * (target_mean - mean_c2_init) + np.random.randn(D) * 0.1
        
        var_c0 = np.abs(np.random.rand(D)) * (1.0 - convergence_factor * 0.3)
        var_c1 = np.abs(np.random.rand(D)) * (1.2 - convergence_factor * 0.3)
        var_c2 = np.abs(np.random.rand(D)) * (0.8 - convergence_factor * 0.2)
        
        samples_c0 = mean_c0 + np.sqrt(var_c0) * np.random.randn(N, D)
        samples_c1 = mean_c1 + np.sqrt(var_c1) * np.random.randn(N, D)
        samples_c2 = mean_c2 + np.sqrt(var_c2) * np.random.randn(N, D)
        
        samples_per_round.append([samples_c0, samples_c1, samples_c2])
    
    plot_prototype_evolution(samples_per_round, num_clients=3, show_samples=True, sample_alpha=0.08)


if __name__ == "__main__":
    # Example 1: Plot with synthetic data
    # print("Plotting synthetic data evolution...")
    # plot_with_synthetic_data()
    
    # Example 2: Load real data from files
    # Uncomment and adjust the path as needed
    # base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CD_trajectory_del_s1/Fold0"
    # base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/art_CFCD_sp1_s3/Fold0"
    # base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/art_CFCDID_sp1_s3/Fold0"
    # base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/MG_08_sp1_s3/Fold0"
    # base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/art_CFCD_no_weighted_training_7_3_sp1_s3/Fold1"
    base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/art_CFCDID_no_weighted_training_sp1_s3/Fold1"
    rounds_to_plot = list(range(1, 41))  # Rounds 1-10
    samples_per_round = load_features_from_rounds(base_dir, num_clients=3, rounds=rounds_to_plot)
    if len(samples_per_round) > 0:
        print(f"Loaded {len(samples_per_round)} rounds")
        plot_prototype_evolution(samples_per_round, num_clients=3, show_samples=True, sample_alpha=0.4)
    else:
        print("No data loaded. Check file paths.")
