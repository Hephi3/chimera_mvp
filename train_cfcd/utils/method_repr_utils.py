import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import math
import os

class PrototypeRepr:    
    def __init__(self, cd_mean, cd_var, wsi_l3_mean, wsi_l3_var, wsi_l2_mean, wsi_l2_var, wsi_l1_mean, wsi_l1_var, num_l3_patches, num_l2_patches, num_l1_patches, label=None):
        self.cd_mean = cd_mean
        self.cd_var = cd_var
        
        self.wsi_l3_mean = wsi_l3_mean
        self.wsi_l3_var = wsi_l3_var
        self.wsi_l2_mean = wsi_l2_mean
        self.wsi_l2_var = wsi_l2_var
        self.wsi_l1_mean = wsi_l1_mean
        self.wsi_l1_var = wsi_l1_var
        
        self.num_l3_patches = num_l3_patches
        self.num_l2_patches = num_l2_patches
        self.num_l1_patches = num_l1_patches
        
        self.label = label
    
    @classmethod
    def from_data(cls, cd_data, wsi_l3_data, wsi_l2_data, wsi_l1_data, label=None):#, plot=False):
        # Calculate patch counts
        lens_l3 = [len(d) for d in wsi_l3_data]
        lens_l2 = [len(d) for d in wsi_l2_data]
        lens_l1 = [len(d) for d in wsi_l1_data]
        num_l3_patches = sum(lens_l3)//len(lens_l3) if len(lens_l3) > 0 else 0
        num_l2_patches = sum(lens_l2)//len(lens_l2) if len(lens_l2) > 0 else 0
        num_l1_patches = sum(lens_l1)//len(lens_l1) if len(lens_l1) > 0 else 0
        
        # Process CD data (each entry is a single vector)
        cd_data_np = np.stack([x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in cd_data])
        cd_mean = np.mean(cd_data_np, axis=0).reshape(-1)
        cd_var = np.var(cd_data_np, axis=0, ddof=1).reshape(-1)
        
        # Process WSI data (each entry is a list of patches, flatten all patches together)
        wsi_l3_flat = [patch.cpu().detach().numpy() if isinstance(patch, torch.Tensor) else patch for sample in wsi_l3_data for patch in sample]
        wsi_l2_flat = [patch.cpu().detach().numpy() if isinstance(patch, torch.Tensor) else patch for sample in wsi_l2_data for patch in sample]
        wsi_l1_flat = [patch.cpu().detach().numpy() if isinstance(patch, torch.Tensor) else patch for sample in wsi_l1_data for patch in sample]
        
        wsi_l3_np = np.stack(wsi_l3_flat)
        wsi_l2_np = np.stack(wsi_l2_flat)
        wsi_l1_np = np.stack(wsi_l1_flat)
        
        wsi_l3_mean = np.mean(wsi_l3_np, axis=0).reshape(-1)
        wsi_l3_var = np.var(wsi_l3_np, axis=0, ddof=1).reshape(-1)
        wsi_l2_mean = np.mean(wsi_l2_np, axis=0).reshape(-1)
        wsi_l2_var = np.var(wsi_l2_np, axis=0, ddof=1).reshape(-1)
        wsi_l1_mean = np.mean(wsi_l1_np, axis=0).reshape(-1)
        wsi_l1_var = np.var(wsi_l1_np, axis=0, ddof=1).reshape(-1)
        
        return cls(cd_mean, cd_var, wsi_l3_mean, wsi_l3_var, wsi_l2_mean, wsi_l2_var, wsi_l1_mean, wsi_l1_var, num_l3_patches, num_l2_patches, num_l1_patches, label=label)


        # if plot:
        #     plot_data_and_prototype(data_np, cls(mean, var, label=label))
        
        # return cls(mean, var, label=label)
    
    @classmethod
    def from_prototypes(cls, prototypes, weights: list[float] = None, plot=False):
        cd_means = [proto.cd_mean for proto in prototypes]
        cd_vars = [proto.cd_var for proto in prototypes]
        wsi_l3_means = [proto.wsi_l3_mean for proto in prototypes]
        wsi_l3_vars = [proto.wsi_l3_var for proto in prototypes]
        wsi_l2_means = [proto.wsi_l2_mean for proto in prototypes]
        wsi_l2_vars = [proto.wsi_l2_var for proto in prototypes]
        wsi_l1_means = [proto.wsi_l1_mean for proto in prototypes]
        wsi_l1_vars = [proto.wsi_l1_var for proto in prototypes]
        num_l3_patches = [proto.num_l3_patches for proto in prototypes]
        num_l2_patches = [proto.num_l2_patches for proto in prototypes]
        num_l1_patches = [proto.num_l1_patches for proto in prototypes]
        labels = [proto.label for proto in prototypes]
        
        if weights is None:
            weights = [1.0 / len(prototypes)] * len(prototypes)
        else:
            sum_weights = sum(weights)
            if sum_weights != 1.0:
                weights = [w / sum_weights for w in weights]

        cd_mean = np.average(cd_means, axis=0, weights=weights).reshape(-1)
        cd_var = np.average(cd_vars, axis=0, weights=weights).reshape(-1)
        wsi_l3_mean = np.average(wsi_l3_means, axis=0, weights=weights).reshape(-1)
        wsi_l3_var = np.average(wsi_l3_vars, axis=0, weights=weights).reshape(-1)
        wsi_l2_mean = np.average(wsi_l2_means, axis=0, weights=weights).reshape(-1)
        wsi_l2_var = np.average(wsi_l2_vars, axis=0, weights=weights).reshape(-1)
        wsi_l1_mean = np.average(wsi_l1_means, axis=0, weights=weights).reshape(-1)
        wsi_l1_var = np.average(wsi_l1_vars, axis=0, weights=weights).reshape(-1)
        num_l3_patches = int(np.average(num_l3_patches, axis=0, weights=weights))
        num_l2_patches = int(np.average(num_l2_patches, axis=0, weights=weights))
        num_l1_patches = int(np.average(num_l1_patches, axis=0, weights=weights))

        label = labels[0] if all(l == labels[0] for l in labels) else None
        
        # if plot:
        #     plot_global_prototype_and_prototypes(cls(mean, var, label=label), prototypes)

        return cls(cd_mean, cd_var, wsi_l3_mean, wsi_l3_var, wsi_l2_mean, wsi_l2_var, wsi_l1_mean, wsi_l1_var, num_l3_patches, num_l2_patches, num_l1_patches, label=label)

    def adapt_towards(self, other_prototype, adaptation_rate: float):
        self.cd_mean = (1 - adaptation_rate) * self.cd_mean + adaptation_rate * other_prototype.cd_mean
        self.cd_var = (1 - adaptation_rate) * self.cd_var + adaptation_rate * other_prototype.cd_var
        self.wsi_l3_mean = (1 - adaptation_rate) * self.wsi_l3_mean + adaptation_rate * other_prototype.wsi_l3_mean
        self.wsi_l3_var = (1 - adaptation_rate) * self.wsi_l3_var + adaptation_rate * other_prototype.wsi_l3_var
        self.wsi_l2_mean = (1 - adaptation_rate) * self.wsi_l2_mean + adaptation_rate * other_prototype.wsi_l2_mean
        self.wsi_l2_var = (1 - adaptation_rate) * self.wsi_l2_var + adaptation_rate * other_prototype.wsi_l2_var
        self.wsi_l1_mean = (1 - adaptation_rate) * self.wsi_l1_mean + adaptation_rate * other_prototype.wsi_l1_mean
        self.wsi_l1_var = (1 - adaptation_rate) * self.wsi_l1_var + adaptation_rate * other_prototype.wsi_l1_var
        self.num_l3_patches = int((1 - adaptation_rate) * self.num_l3_patches + adaptation_rate * other_prototype.num_l3_patches)
        self.num_l2_patches = int((1 - adaptation_rate) * self.num_l2_patches + adaptation_rate * other_prototype.num_l2_patches)
        self.num_l1_patches = int((1 - adaptation_rate) * self.num_l1_patches + adaptation_rate * other_prototype.num_l1_patches)

    def distance_point(self, dp_cd, dp_wsi_l3, dp_wsi_l2, dp_wsi_l1):
        distances = []
        
        z_score_cds = (dp_cd - self.cd_mean) / np.sqrt(self.cd_var + 1e-12)
        z_score_l3 = (dp_wsi_l3 - self.wsi_l3_mean) / np.sqrt(self.wsi_l3_var + 1e-12)
        z_score_l2 = (dp_wsi_l2 - self.wsi_l2_mean) / np.sqrt(self.wsi_l2_var + 1e-12)
        z_score_l1 = (dp_wsi_l1 - self.wsi_l1_mean) / np.sqrt(self.wsi_l1_var + 1e-12)
        
        distances.append(np.mean(np.abs(z_score_cds)))
        distances.append(np.mean(np.abs(z_score_l3)))
        distances.append(np.mean(np.abs(z_score_l2)))
        distances.append(np.mean(np.abs(z_score_l1)))
        
        # means = [self.cd_mean, self.wsi_l3_mean, self.wsi_l2_mean, self.wsi_l1_mean]
        # vars = [self.cd_var, self.wsi_l3_var, self.wsi_l2_var, self.wsi_l1_var]
        # distances = []
        # for mean, var in zip(means, vars):
        #      # Use z-score
        #     z_scores = (data_point - mean) / np.sqrt(var + 1e-12)
        #     # return np.linalg.norm(z_scores)
        #     mean_distance = np.mean(np.abs(z_scores))
        #     distances.append(mean_distance)
        return np.mean(distances) #TODO: Could be weighted. Does l3 have more information or less then l1?
    
    def weight_point(self, dp_cd, dp_wsi_l3, dp_wsi_l2, dp_wsi_l1, strictness=1.0):
        distance = self.distance_point(dp_cd, dp_wsi_l3, dp_wsi_l2, dp_wsi_l1)
        weight = math.exp(strictness* -distance**2)  # Example: exponential decay based on distance
        return weight
    
    def distance_prototype(self, other_prototype):
        # KL divergence between two Gaussians maybe not a good idea as it is not a metric (not symmetric)
        # Wasserstein distance between two Gaussians?!
        means = [(self.cd_mean, other_prototype.cd_mean),
                 (self.wsi_l3_mean, other_prototype.wsi_l3_mean),
                 (self.wsi_l2_mean, other_prototype.wsi_l2_mean),
                 (self.wsi_l1_mean, other_prototype.wsi_l1_mean)]
        vars = [(self.cd_var, other_prototype.cd_var),
                (self.wsi_l3_var, other_prototype.wsi_l3_var),
                (self.wsi_l2_var, other_prototype.wsi_l2_var),
                (self.wsi_l1_var, other_prototype.wsi_l1_var)]
        distances = []
        for (mean1, mean2), (var1, var2) in zip(means, vars):
            squared_distances = np.sum(mean1 - mean2) ** 2 + np.sum(var1 + var2 - 2 * np.sqrt(var1 * var2))
            distances.append(np.sqrt(squared_distances))
        return np.mean(distances)
        
        
        
    def print_info(self):
        print("Prototype Info:")
        print(f" Mean CD: {self.cd_mean}, Var: {self.cd_var}")
        print(f" Mean WSI L3: {self.wsi_l3_mean}, Var: {self.wsi_l3_var}")
        print(f" Mean WSI L2: {self.wsi_l2_mean}, Var: {self.wsi_l2_var}")
        print(f" Mean WSI L1: {self.wsi_l1_mean}, Var: {self.wsi_l1_var}")
        if self.label is not None:
            print(f" Label: {self.label}")

    def serialize(self):
        return json.dumps({
            "cd_mean": self.cd_mean.tolist(),
            "cd_var": self.cd_var.tolist(),
            "wsi_l3_mean": self.wsi_l3_mean.tolist(),
            "wsi_l3_var": self.wsi_l3_var.tolist(),
            "wsi_l2_mean": self.wsi_l2_mean.tolist(),
            "wsi_l2_var": self.wsi_l2_var.tolist(),
            "wsi_l1_mean": self.wsi_l1_mean.tolist(),
            "wsi_l1_var": self.wsi_l1_var.tolist(),
            "num_l3_patches": self.num_l3_patches,
            "num_l2_patches": self.num_l2_patches,
            "num_l1_patches": self.num_l1_patches,
            "label": self.label
        })
    
    @classmethod
    def deserialize(cls, json_str):
        data = json.loads(json_str)
        cd_mean = np.array(data["cd_mean"])
        cd_var = np.array(data["cd_var"])
        wsi_l3_mean = np.array(data["wsi_l3_mean"])
        wsi_l3_var = np.array(data["wsi_l3_var"])
        wsi_l2_mean = np.array(data["wsi_l2_mean"])
        wsi_l2_var = np.array(data["wsi_l2_var"])
        wsi_l1_mean = np.array(data["wsi_l1_mean"])
        wsi_l1_var = np.array(data["wsi_l1_var"])
        num_l3_patches = data["num_l3_patches"]
        num_l2_patches = data["num_l2_patches"]
        num_l1_patches = data["num_l1_patches"]
        label = data["label"]
        return cls(cd_mean, cd_var, wsi_l3_mean, wsi_l3_var, wsi_l2_mean, wsi_l2_var, wsi_l1_mean, wsi_l1_var, num_l3_patches, num_l2_patches, num_l1_patches, label=label)

    def sample(self, num_samples=1, variance_scale=1.0):
        # CD:
        cd_samples = np.random.normal(loc=self.cd_mean, scale=np.sqrt(self.cd_var) * variance_scale, size=(num_samples, len(self.cd_mean)))
        # WSI L3:
        wsi_l3_samples = np.random.normal(loc=self.wsi_l3_mean, scale=np.sqrt(self.wsi_l3_var) * variance_scale, size=(num_samples, self.num_l3_patches, len(self.wsi_l3_mean)))
        # WSI L2:
        wsi_l2_samples = np.random.normal(loc=self.wsi_l2_mean, scale=np.sqrt(self.wsi_l2_var) * variance_scale, size=(num_samples, self.num_l2_patches, len(self.wsi_l2_mean)))
        # WSI L1:
        wsi_l1_samples = np.random.normal(loc=self.wsi_l1_mean, scale=np.sqrt(self.wsi_l1_var) * variance_scale, size=(num_samples, self.num_l1_patches, len(self.wsi_l1_mean)))

        # Reorder them to samples of (CD, WSI L3 patches, WSI L2 patches, WSI L1 patches)
        samples = []
        for i in range(num_samples):
            sample = {
                "cd": cd_samples[i],
                "wsi_l3": wsi_l3_samples[i],
                "wsi_l2": wsi_l2_samples[i],
                "wsi_l1": wsi_l1_samples[i]
            }
            samples.append(sample)
        return samples

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