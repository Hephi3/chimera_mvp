import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


def plot_prototype_test(samples, show_samples=True, sample_alpha=0.05):
    """
    Plot prototypes per class for each client.
    
    Args:
        samples: Dict structure where samples[client_id][class_label] contains features
        show_samples: Whether to show individual data samples (default True)
        sample_alpha: Alpha value for individual samples (default 0.05)
    """
    # Create prototypes per class for each client
    protos = [
        [Prototype.from_data(samples[client_id][brs]) for brs in [0,1,2]] for client_id in range(3)
    ]

    # Create global prototype
    protos_flattened = [proto for client_protos in protos for proto in client_protos]
    proto_global = Prototype.from_prototypes(protos_flattened)
    
    samples_flattened = [samples[client_id][brs] for client_id in range(3) for brs in [0,1,2]]

    # Fit PCA on all data
    all_for_pca = np.vstack(samples_flattened)
    pca = PCA(n_components=2)
    pca.fit(all_for_pca)

    # Define colors and markers
    client_colors = ['blue', 'green', 'orange']
    class_markers = ['o', 's', '^']
    class_names = ['BRS 0', 'BRS 1', 'BRS 2']

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))

    # First pass: Plot individual samples in background (if enabled)
    if show_samples:
        for client_idx in range(3):
            for class_idx in [0, 1, 2]:
                data_2d = pca.transform(samples[client_idx][class_idx])
                ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                          c=client_colors[client_idx], marker=class_markers[class_idx],
                          s=50, alpha=sample_alpha, edgecolors='none', zorder=0)

    # Second pass: Plot prototypes and their variance
    # Plot global prototype first
    proto_global_2d = pca.transform(proto_global.mean.reshape(1, -1))
    ax.scatter(proto_global_2d[:, 0], proto_global_2d[:, 1], 
              c='red', marker='X', s=150, alpha=1.0, 
              edgecolors='black', linewidths=1, zorder=3, label='Global Prototype')
    
    # Plot global prototype variance ellipse
    cov_128 = np.diag(proto_global.var)
    cov_2d = pca.components_ @ cov_128 @ pca.components_.T
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
    ellipse = Ellipse(xy=proto_global_2d[0], width=width, height=height, angle=theta,
                     edgecolor='red', fc='None', lw=1.5, alpha=0.4, zorder=2)
    ax.add_patch(ellipse)

    # Plot each client's prototypes by class
    legend_added = {'client': [False, False, False]}
    
    for client_idx in range(3):
        print(f"Visualizing Client {client_idx} data and prototype...")
        for class_idx in [0, 1, 2]:
            proto_2d = pca.transform(protos[client_idx][class_idx].mean.reshape(1, -1))

            # Create labels only for first occurrence
            client_label = f'Client {client_idx + 1}' if not legend_added['client'][client_idx] else None
            
            # Plot prototype mean
            ax.scatter(proto_2d[:, 0], proto_2d[:, 1], 
                      c=client_colors[client_idx], marker=class_markers[class_idx],
                      s=150, alpha=1.0, edgecolors='black', linewidths=1.5, zorder=2,
                      label=client_label)
            
            # Mark that we've added this client's legend
            if client_label:
                legend_added['client'][client_idx] = True

            # Plot variance ellipse
            cov_128 = np.diag(protos[client_idx][class_idx].var)
            cov_2d = pca.components_ @ cov_128 @ pca.components_.T
            vals, vecs = np.linalg.eigh(cov_2d)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
            ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta,
                            edgecolor=client_colors[client_idx], fc='None', lw=1.5, alpha=0.5, zorder=1)
            ax.add_patch(ellipse)
    
    # Add dummy scatter points for class marker legend
    for class_idx in [0, 1, 2]:
        ax.scatter([], [], marker=class_markers[class_idx], c='gray', s=100,
                  label=class_names[class_idx], edgecolors='black', linewidths=1)

    ax.set_xlabel("PC 1", fontsize=18)
    ax.set_ylabel("PC 2", fontsize=18)
    ax.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("prototype_per_class.png", dpi=300)
    print("Plot saved as prototype_per_class.png")
    plt.show()


def plot_with_random_values():
    np.random.seed(42)
    N = 100
    D = 128

    # Client 0: ID: 0

    mean_c0 = np.random.randn(D) * 1
    var_c0 = np.abs(np.random.rand(D)) * 1
    samples_c0 = mean_c0 + np.sqrt(var_c0) * np.random.randn(N, D)  # N samples in D dimensions

    # Client 1: ID: 1
    mean_c1 = mean_c0 + 0.2 + np.random.randn(D) * 0.1
    var_c1 = np.abs(np.random.rand(D)) * 1.2
    samples_c1 = mean_c1 + np.sqrt(var_c1) * np.random.randn(N, D)  

    # Client 2: ID: 2
    mean_c2 = mean_c0 -0.5 + np.random.randn(D) * -0.1
    var_c2 = np.abs(np.random.rand(D)) * 0.5
    samples_c2 = mean_c2 + np.sqrt(var_c2) * np.random.randn(N, D)

    # samples_c0 = 0 + 1 * np.random.randn(N, D)  # N samples in D dimensions
    # samples_c1 = 2 + 2 * np.random.randn(N, D)
    # samples_c2 = -1 + 0.5 * np.random.randn(N, D)
    print(np.array(samples_c0).shape)
    plot_prototype_test(samples_c0, samples_c1, samples_c2)

if __name__ == "__main__":
    np.random.seed(42)
    # plot_with_random_values()
    
    # dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CD_del_s1/Fold0/"
    # dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CDID_del_s1/Fold0/"
    # dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CDID_no_train_del_s1/Fold0/"
    # dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CD_no_train_del_s1/Fold0/"
    dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/art_CFCD_no_weighted_training_7_3_sp1_s1/Fold4"
    # dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CDID_gradient_del_s1/Fold0"
    samples = []
    for i in [0,1,2]:
        data = np.load(f"{dir}/debug_features_client_{i}_round_40.npz")
        features = data["features"]
        labels = data["labels"]
        samples.append({0: np.array(features[labels==0]).squeeze(1), 1: np.array(features[labels==1]).squeeze(1), 2: np.array(features[labels==2]).squeeze(1)})
        print("Loaded client", i, "with features shape", features.shape, "and labels shape", labels.shape, "and per class samples:", [len(features) for features in samples[-1].values()])

    plot_prototype_test(samples, show_samples=True, sample_alpha=0.4)