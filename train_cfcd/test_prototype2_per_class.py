import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


def plot_prototype_test(samples):

    # Create prototypes
    protos = [
        [Prototype.from_data(samples[client_id][brs]) for brs in [0,1,2]] for client_id in range(3)
    ]


    # num_samples = 15
    # variance_scale = 0.6
    # sampled_c0 = proto_c0.sample(num_samples=num_samples, variance_scale=variance_scale)
    # sampled_c1 = proto_c1.sample(num_samples=num_samples, variance_scale=variance_scale)
    # sampled_c2 = proto_c2.sample(num_samples=num_samples, variance_scale=variance_scale)


    # Create global prototype
    protos_flattened = [proto for client_protos in protos for proto in client_protos]
    proto_global = Prototype.from_prototypes(protos_flattened)
    
    samples_flattened = [samples[client_id][brs] for client_id in range(3) for brs in [0,1,2]]


    all_for_pca = np.vstack(samples_flattened)
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_for_pca)

    # samples = [samples_c0, samples_c1, samples_c2]
    # prototypes = [proto_c0, proto_c1, proto_c2]
    colors = ['blue', 'green', 'orange']
    markers = ['o', 's', '^']
    # sampled = [sampled_c0, sampled_c1, sampled_c2]

    plt.figure(figsize=(15, 8))


    # plot global prototype
    proto_global_2d = pca.transform(proto_global.mean.reshape(1, -1))
    plt.scatter(proto_global_2d[:, 0], proto_global_2d[:, 1], c='black', marker='x', s=200, label='Global Prototype Mean')
    # wasserstein_distances = [proto_global.distance_prototype(proto_c0),
    #                             proto_global.distance_prototype(proto_c1),
    #                             proto_global.distance_prototype(proto_c2)]

    for i in range(3):

        print(f"Visualizing Client {i} data and prototype...")
        for brs in [0,1,2]:
            data_2d = pca.transform(samples[i][brs])
            proto_2d = pca.transform(protos[i][brs].mean.reshape(1, -1))

            plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, color=colors[i], label=f'Client {i} Data BRS {brs}', marker=markers[brs])
            plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c=colors[i], marker=markers[brs], s=200, label='Prototype Mean')
            # sampled_2d = pca.transform(sampled[i])
            # plt.scatter(sampled_2d[:, 0], sampled_2d[:, 1], c=colors[i], marker='x', label=f'Client {i} Samples')

            # Plot prototype variance as ellipse
            cov_128 = np.diag(protos[i][brs].var)  # (128, 128)
            cov_2d = pca.components_ @ cov_128 @ pca.components_.T  # (2, 2)
            vals, vecs = np.linalg.eigh(cov_2d)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
            ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta,
                            edgecolor=colors[i], fc='None', lw=2, label='Prototype Var (2σ)')
            plt.gca().add_patch(ellipse)
            
            # Plot wasserstein distance lines to global prototype
            # plt.plot([proto_2d[0, 0], proto_global_2d[0, 0]], [proto_2d[0, 1], proto_global_2d[0, 1]], 'k--', label='Distance to Global Prototype')
            # plt.text((proto_2d[0, 0] + proto_global_2d[0, 0]) / 2, (proto_2d[0, 1] + proto_global_2d[0, 1]) / 2, f"{wasserstein_distances[i]:.2f}", color='black')



    # Plot prototype variance as ellipse
    cov_128 = np.diag(proto_global.var)  # (128, 128)
    cov_2d = pca.components_ @ cov_128 @ pca.components_.T  # (2, 2)
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
    ellipse = Ellipse(xy=proto_global_2d[0], width=width, height=height, angle=theta,
                    edgecolor='black', fc='None', lw=2, label='Global Prototype Var (2σ)')
    plt.gca().add_patch(ellipse)


    plt.title(f"Data and Prototype Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.5, 1))
    plt.tight_layout()
    plt.grid()
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
    dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CD_trajectory_del_s1/Fold0"
    # dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/results/CDID_gradient_del_s1/Fold0"
    samples = []
    for i in [0,1,2]:
        data = np.load(f"{dir}/debug_features_client_{i}_round_0.npz")
        features = data["features"]
        labels = data["labels"]
        samples.append({0: np.array(features[labels==0]).squeeze(1), 1: np.array(features[labels==1]).squeeze(1), 2: np.array(features[labels==2]).squeeze(1)})
        print("Loaded client", i, "with features shape", features.shape, "and labels shape", labels.shape, "and per class samples:", [len(features) for features in samples[-1].values()])
        # samples.append((features, labels))

    # print(np.array(samples[0]).squeeze(1))
    # # print("Sample 1 of client 1:", samples[0][0])
    plot_prototype_test(samples)