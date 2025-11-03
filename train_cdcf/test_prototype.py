import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# 1. Generate synthetic data (3 clusters in 128D)
np.random.seed(42)
N = 100
D = 128
means = [np.random.randn(D) * 2 for _ in range(3)]
vars = [np.abs(np.random.rand(D)) + 0.5 for _ in range(3)]

clusters = []
for m, v in zip(means, vars):
    data = np.random.normal(loc=m, scale=np.sqrt(v), size=(N, D))
    clusters.append(data)

# 2. Create Prototype objects for each cluster
prototypes = [Prototype.from_data([d for d in cluster], label=f"cluster_{i}") for i, cluster in enumerate(clusters)]

# 3. Create a global prototype from all clusters
all_data = np.vstack(clusters)
global_prototype = Prototype.from_data([d for d in all_data], label="global")


# 4. Visualize one cluster and its prototype, with samples and distance
print("Visualizing cluster 0 and its prototype, with samples and distances...")
# Prepare data for PCA
data_np = clusters[0]
proto_mean = prototypes[0].mean.reshape(1, -1)
samples = prototypes[0].sample(num_samples=4)
sample_point = data_np[0]

# Fit PCA on cluster data + prototype mean + samples
all_for_pca = np.vstack([data_np, proto_mean, samples, prototypes[1].mean.reshape(1, -1)])
pca = PCA(n_components=2)
all_2d = pca.fit_transform(all_for_pca)
data_2d = all_2d[:len(data_np)]
proto_2d = all_2d[len(data_np):len(data_np)+1]
samples_2d = all_2d[len(data_np)+1:len(data_np)+6]
proto1_2d = all_2d[-1:]
sample_point_2d = data_2d[0]

plt.figure(figsize=(9, 7))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', label='Cluster 0 Data', alpha=0.5)
plt.scatter(proto_2d[0, 0], proto_2d[0, 1], c='red', marker='x', s=200, label='Prototype 0 Mean')
plt.scatter(samples_2d[:, 0], samples_2d[:, 1], c='green', marker='o', s=80, label='Samples from Prototype 0')

# Draw line from sample_point to prototype mean
plt.plot([sample_point_2d[0], proto_2d[0, 0]], [sample_point_2d[1], proto_2d[0, 1]], 'k--', label='Sample-to-Prototype Dist')

# Draw line between prototype 0 and prototype 1 means
plt.plot([proto_2d[0, 0], proto1_2d[0, 0]], [proto_2d[0, 1], proto1_2d[0, 1]], 'm-.', label='Prototype 0-1 Dist')

# Plot prototype variance as ellipse
cov_128 = np.diag(prototypes[0].var)
cov_2d = pca.components_ @ cov_128 @ pca.components_.T
vals, vecs = np.linalg.eigh(cov_2d)
order = vals.argsort()[::-1]
vals = vals[order]
vecs = vecs[:, order]
theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta, edgecolor='red', fc='None', lw=2, label='Prototype 0 Var (2σ)')
plt.gca().add_patch(ellipse)

plt.legend()
plt.title("Cluster 0, Prototype, Samples, and Distances in PCA 2D Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# 5. Visualize all prototypes and the global prototype
print("Visualizing all prototypes and the global prototype...")
from utils.method_utils import plot_global_prototype_and_prototypes
plot_global_prototype_and_prototypes(global_prototype, prototypes)

# 6. Demonstrate distance_point and distance_prototype
dist_point = prototypes[0].distance_point(sample_point)
print(f"Distance from sample point to its prototype: {dist_point:.4f}")

dist_proto = prototypes[0].distance_prototype(prototypes[1])
print(f"Wasserstein-like distance between prototype 0 and 1: {dist_proto:.4f}")

# 7. Demonstrate sampling from a prototype (already visualized above)
print("5 samples from prototype 0:")
print(samples)
