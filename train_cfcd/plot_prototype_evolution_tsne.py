import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.manifold import TSNE


def plot_prototype_evolution_tsne(samples_per_round, num_clients=3, client_colors=None, show_samples=True, sample_alpha=0.05, perplexity=30, random_state=42):
    """
    Plot the evolution of prototypes over multiple rounds using t-SNE.
    
    Note: Variance ellipses are not included since t-SNE is a non-linear transformation
    and variance in the original space cannot be meaningfully projected.
    
    Args:
        samples_per_round: List of lists, where samples_per_round[round][client] is the feature array for that client at that round
        num_clients: Number of clients (default 3)
        client_colors: List of colors for each client (default ['blue', 'green', 'orange'])
        show_samples: Whether to show individual data samples (default True)
        sample_alpha: Alpha value for individual samples (default 0.05)
        perplexity: t-SNE perplexity parameter (default 30)
        random_state: Random seed for reproducibility (default 42)
    """
    if client_colors is None:
        client_colors = ['blue', 'green', 'orange']
    
    num_rounds = len(samples_per_round)
    
    # Collect all data and prototype means for t-SNE fitting
    all_data = []
    prototype_means = []
    global_proto_means = []
    
    for round_samples in samples_per_round:
        for client_samples in round_samples:
            all_data.append(client_samples)
        
        # Compute prototypes for this round
        prototypes = [Prototype.from_data(round_samples[i]) for i in range(num_clients)]
        global_proto = Prototype.from_prototypes(prototypes)
        
        # Store prototype means
        for proto in prototypes:
            prototype_means.append(proto.mean)
        global_proto_means.append(global_proto.mean)
    
    all_data = np.vstack(all_data)
    prototype_means = np.vstack(prototype_means)
    global_proto_means = np.vstack(global_proto_means)
    
    # Combine all points (samples + prototypes + global prototypes) for t-SNE
    combined_data = np.vstack([all_data, prototype_means, global_proto_means])
    
    # Fit t-SNE on all data
    print(f"Running t-SNE on {combined_data.shape[0]} points...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000)
    embedded = tsne.fit_transform(combined_data)
    
    # Split embedded data back
    n_samples = all_data.shape[0]
    n_protos = prototype_means.shape[0]
    
    samples_embedded = embedded[:n_samples]
    protos_embedded = embedded[n_samples:n_samples + n_protos]
    global_protos_embedded = embedded[n_samples + n_protos:]
    
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
        sample_idx = 0
        for round_idx, round_samples in enumerate(samples_per_round):
            round_alpha = alphas[round_idx] * sample_alpha
            round_size = sizes[round_idx]
            for client_idx in range(num_clients):
                n_client_samples = round_samples[client_idx].shape[0]
                samples_2d = samples_embedded[sample_idx:sample_idx + n_client_samples]
                sample_idx += n_client_samples
                
                # Plot samples as dots with increasing size for newer rounds
                ax.scatter(samples_2d[:, 0], samples_2d[:, 1], 
                          c=client_colors[client_idx], s=round_size, alpha=round_alpha, 
                          edgecolors='none', zorder=0)
    
    # Second pass: Plot prototypes (no variance ellipses for t-SNE)
    proto_idx = 0
    for round_idx in range(num_rounds):
        alpha = alphas[round_idx]
        
        # Plot global prototype for this round
        global_label = 'Global Prototype' if round_idx == num_rounds - 1 else None
        ax.scatter(global_protos_embedded[round_idx, 0], global_protos_embedded[round_idx, 1], 
                  c='black', marker='X', s=150, alpha=alpha*0.3, edgecolors='red', 
                  linewidths=1, zorder=1, label=global_label)
        
        # Plot each client
        for client_idx in range(num_clients):
            proto_2d = protos_embedded[proto_idx]
            client_proto_positions[client_idx].append(proto_2d)
            proto_idx += 1
            
            # Plot prototype mean
            label = f'Client {client_idx + 1}' if round_idx == num_rounds - 1 else None
            ax.scatter(proto_2d[0], proto_2d[1], c=client_colors[client_idx], 
                      marker='o', s=100, alpha=alpha, label=label, edgecolors='black', 
                      linewidths=1, zorder=2)
    
    # Draw trajectory lines for each client
    for client_idx in range(num_clients):
        if len(client_proto_positions[client_idx]) > 1:
            positions = np.array(client_proto_positions[client_idx])
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=client_colors[client_idx], linestyle='--', alpha=0.6, linewidth=2, zorder=1)
            
            # Add arrows to show direction
            for i in range(len(positions) - 1):
                dx = positions[i+1, 0] - positions[i, 0]
                dy = positions[i+1, 1] - positions[i, 1]
                # Calculate arrow size based on distance
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 0:
                    head_width = distance * 0.15
                    head_length = distance * 0.12
                    ax.arrow(positions[i, 0], positions[i, 1], 
                            dx*0.7, dy*0.7, head_width=head_width, head_length=head_length, 
                            fc=client_colors[client_idx], ec=client_colors[client_idx], 
                            alpha=0.4, zorder=1)
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=18)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=18)
    ax.legend(loc='upper right', fontsize=18, bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("prototype_evolution_tsne.png")
    print("Plot saved as prototype_evolution_tsne.png")


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
    
    plot_prototype_evolution_tsne(samples_per_round, num_clients=3, show_samples=True, sample_alpha=0.08, perplexity=30)


if __name__ == "__main__":
    # Example 1: Plot with synthetic data
    # print("Plotting synthetic data evolution with t-SNE...")
    # plot_with_synthetic_data()
    
    # Example 2: Load real data from files
    # Uncomment and adjust the path as needed
    base_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/art_CFCD_sp1_s3/Fold0"
    rounds_to_plot = list(range(1, 41))  # Rounds 1-40
    samples_per_round = load_features_from_rounds(base_dir, num_clients=3, rounds=rounds_to_plot)
    if len(samples_per_round) > 0:
        print(f"Loaded {len(samples_per_round)} rounds")
        plot_prototype_evolution_tsne(samples_per_round, num_clients=3, show_samples=True, sample_alpha=0.4, perplexity=30)
    else:
        print("No data loaded. Check file paths.")
