import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import math
import os

class Prototype:    
    def __init__(self, mean, var, label=None):
        self.mean = mean
        self.var = var
        self.label = label
    
    @classmethod
    def from_data(cls, data, label=None, plot=False):
        data_np = np.stack([x.cpu().detach().numpy().reshape(-1) if isinstance(x, torch.Tensor) else x for x in data])
        mean = np.mean(data_np, axis=0).reshape(-1)
        # Handle case when we have very few samples
        if len(data_np) <= 1:
            raise ValueError("Not enough data points to compute variance.")
        else:
            var = np.var(data_np, axis=0, ddof=1).reshape(-1)
            # Replace any NaN variances with a small positive value
            var = np.nan_to_num(var, nan=0.01)

        if plot:
            plot_data_and_prototype(data_np, cls(mean, var, label=label))
        
        return cls(mean, var, label=label)
    
    @classmethod
    def from_prototypes(cls, prototypes, weights: list[float] = None, plot=False):
        means = [proto.mean for proto in prototypes]
        vars = [proto.var for proto in prototypes]
        labels = [proto.label for proto in prototypes]
        if weights is None:
            weights = [1.0 / len(prototypes)] * len(prototypes)
        else:
            sum_weights = sum(weights)
            if sum_weights != 1.0:
                weights = [w / sum_weights for w in weights]

        mean = np.average(means, axis=0, weights=weights).reshape(-1)
        var = np.average(vars, axis=0, weights=weights).reshape(-1)
        label = labels[0] if all(l == labels[0] for l in labels) else None
        
        if plot:
            plot_global_prototype_and_prototypes(cls(mean, var, label=label), prototypes)
        
        return cls(mean, var, label=label)

    def adapt_towards(self, other_prototype, adaptation_rate: float):
        self.mean = (1 - adaptation_rate) * self.mean + adaptation_rate * other_prototype.mean
        self.var = (1 - adaptation_rate) * self.var + adaptation_rate * other_prototype.var

    def distance_point(self, data_point):
        # Use z-score
        z_scores = (data_point - self.mean) / np.sqrt(self.var + 1e-12)
        # return np.linalg.norm(z_scores)
        mean = np.mean(np.abs(z_scores))
        # print("Mean z-score distance:", mean)
        return mean
    
    def weight_point(self, data_point):
        distance = self.distance_point(data_point)
        weight = math.exp(-distance**2)  # Example: exponential decay based on distance
        return weight
    
    def distance_prototype(self, other_prototype):
        # KL divergence between two Gaussians maybe not a good idea as it is not a metric (not symmetric)
        # Wasserstein distance between two Gaussians?!
        squared_distances = np.sum(self.mean - other_prototype.mean) ** 2 + np.sum(self.var + other_prototype.var - 2 * np.sqrt(self.var * other_prototype.var))
        # np.sum(self.mean - other_prototype.mean) ** 2# TODO + np.sum(self.var + other_prototype.var - 2 * np.sqrt(self.var * other_prototype.var))
        distances = np.sqrt(squared_distances)
        return np.mean(distances)
        
        
        

    def print_info(self):
        print("Prototype Info:")
        print(f"Mean mean: {np.mean(self.mean)}, Var mean: {np.mean(self.var)}")
        # print(f" Mean: {self.mean}, Var: {self.var}")
        if self.label is not None:
            print(f" Label: {self.label}")

    def serialize(self):
        return json.dumps({
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "label": self.label
        })
    
    @classmethod
    def deserialize(cls, json_str):
        data = json.loads(json_str)
        mean = np.array(data["mean"])
        var = np.array(data["var"])
        label = data["label"]
        return cls(mean, var, label=label)
    
    def sample(self, num_samples=1, variance_scale=1.0):
        print("Size:", (num_samples, len(self.mean)), "Mean shape:", self.mean.shape, "Var shape:", self.var.shape)
        return np.random.normal(loc=self.mean, scale=np.sqrt(self.var) * variance_scale, size=(num_samples, len(self.mean)))
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.serialize())

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            return cls.deserialize(f.read())


def plot_data_and_prototype(data_np, prototype):
    # data: list of tensors or numpy arrays, shape (N, 128)
    # prototype: Prototype instance with .mean and .var (both shape (128,))
    # Convert data to numpy if needed
    from sklearn.decomposition import PCA
    proto_mean = prototype.mean.reshape(1, -1)  # shape (1, 128)
    
    # Fit PCA on data
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_np)
    proto_2d = pca.transform(proto_mean)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', label='Data', alpha=0.5)
    plt.scatter(proto_2d[0, 0], proto_2d[0, 1], c='red', marker='x', s=200, label='Prototype Mean')
    
    # Optional: plot prototype variance as an ellipse
    # Project the covariance matrix to 2D
    cov_128 = np.diag(prototype.var)
    cov_2d = pca.components_ @ cov_128 @ pca.components_.T
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
    ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta, edgecolor='red', fc='None', lw=2, label='Prototype Var (2σ)')
    plt.gca().add_patch(ellipse)
    
    plt.legend()
    plt.title("Data and Prototype in PCA 2D Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def plot_global_prototype_and_prototypes(global_prototype, prototypes):
    """
    Plot several prototypes and the global prototype in the same PCA 2D space,
    showing variance ellipses for each.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib.patches import Ellipse

    # Stack all prototype means for PCA
    all_means = np.stack([p.mean for p in prototypes] + [global_prototype.mean])
    pca = PCA(n_components=2)
    all_means_2d = pca.fit_transform(all_means)

    # Split back into client and global
    client_means_2d = all_means_2d[:-1]
    global_mean_2d = all_means_2d[-1:]

    plt.figure(figsize=(8, 6))
    # Plot client prototypes
    plt.scatter(client_means_2d[:, 0], client_means_2d[:, 1], c='blue', label='Client Prototypes')

    # Plot variance ellipses for each client prototype
    for i, proto in enumerate(prototypes):
        cov_128 = np.diag(proto.var)
        cov_2d = pca.components_ @ cov_128 @ pca.components_.T
        vals, vecs = np.linalg.eigh(cov_2d)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
        ellipse = Ellipse(xy=client_means_2d[i], width=width, height=height, angle=theta,
                          edgecolor='blue', fc='None', lw=1, alpha=0.5)
        plt.gca().add_patch(ellipse)

    # Plot global prototype
    plt.scatter(global_mean_2d[0, 0], global_mean_2d[0, 1], c='red', marker='x', s=200, label='Global Prototype')

    # Plot global prototype variance as ellipse
    cov_128 = np.diag(global_prototype.var)
    cov_2d = pca.components_ @ cov_128 @ pca.components_.T
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
    ellipse = Ellipse(xy=global_mean_2d[0], width=width, height=height, angle=theta,
                      edgecolor='red', fc='None', lw=2, label='Global Var (2σ)')
    plt.gca().add_patch(ellipse)

    plt.legend()
    plt.title("Client and Global Prototypes in PCA 2D Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
    

# def generate_prototype_of_prototypes():
#     pass

# def generate_prototype_of_data(data):
#     # prototypeBRS1 = {wsi_proto: [...], cd_proto: [...]}
#     # prototypeBRS2 = {wsi_proto: [...], cd_proto: [...]}
#     # prototypeBRS3 = {wsi_proto: [...], cd_proto: [...]}
#     prototype_BRS1 = {
#         "wsi_proto": np.mean([datapoint[0] for datapoint in data if datapoint[1] == 1], axis=0),  # Mean WSI data
#         "cd_proto": np.mean([datapoint[3] for datapoint in data if datapoint[1] == 1], axis=0)  # Mean CD data
#     }
#     prototype_BRS2 = {
#         "wsi_proto": np.mean([datapoint[0] for datapoint in data if datapoint[1] == 2], axis=0),  # Mean WSI data
#         "cd_proto": np.mean([datapoint[3] for datapoint in data if datapoint[1] == 2], axis=0)  # Mean CD data
#     }
#     prototype_BRS3 = {
#         "wsi_proto": np.mean([datapoint[0] for datapoint in data if datapoint[1] == 3], axis=0),  # Mean WSI data
#         "cd_proto": np.mean([datapoint[3] for datapoint in data if datapoint[1] == 3], axis=0)  # Mean CD data
#     }
#     prototype_all = {
#         "wsi_proto": np.mean([datapoint[0] for datapoint in data], axis=0),  # Mean WSI data
#         "cd_proto": np.mean([datapoint[3] for datapoint in data], axis=0)  # Mean CD data
#     }

# python federated_train_method.py --gpus 2 --num_clients 3 --exp_code ID --no_verbose --split_dir chimera_3_5_0.1_1 --folds 5 --seed 1 --num_rounds 10 --no_phases --max_epochs 1