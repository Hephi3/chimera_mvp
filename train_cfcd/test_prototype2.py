import numpy as np
import matplotlib.pyplot as plt
from utils.method_utils import Prototype
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

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

# Create prototypes
proto_c0 = Prototype.from_data(samples_c0)
proto_c1 = Prototype.from_data(samples_c1)
proto_c2 = Prototype.from_data(samples_c2)


num_samples = 15
variance_scale = 0.6
sampled_c0 = proto_c0.sample(num_samples=num_samples, variance_scale=variance_scale)
sampled_c1 = proto_c1.sample(num_samples=num_samples, variance_scale=variance_scale)
sampled_c2 = proto_c2.sample(num_samples=num_samples, variance_scale=variance_scale)


# Create global prototype
proto_global = Prototype.from_prototypes([proto_c0, proto_c1, proto_c2])


all_for_pca = np.vstack([samples_c0, samples_c1, samples_c2])
pca = PCA(n_components=2)
all_2d = pca.fit_transform(all_for_pca)

samples = [samples_c0, samples_c1, samples_c2]
prototypes = [proto_c0, proto_c1, proto_c2]
colors = ['blue', 'green', 'orange']
sampled = [sampled_c0, sampled_c1, sampled_c2]

plt.figure(figsize=(10, 8))


# plot global prototype
proto_global_2d = pca.transform(proto_global.mean.reshape(1, -1))
plt.scatter(proto_global_2d[:, 0], proto_global_2d[:, 1], c='black', marker='x', s=200, label='Global Prototype Mean')
wasserstein_distances = [proto_global.distance_prototype(proto_c0),
                             proto_global.distance_prototype(proto_c1),
                             proto_global.distance_prototype(proto_c2)]

for i in range(3):

    print(f"Visualizing Client {i} data and prototype...")
    data_2d = pca.transform(samples[i])
    proto_2d = pca.transform(prototypes[i].mean.reshape(1, -1))

    plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, color=colors[i], label=f'Client {i} Data')
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c='red', marker='x', s=200, label='Prototype Mean')
    sampled_2d = pca.transform(sampled[i])
    plt.scatter(sampled_2d[:, 0], sampled_2d[:, 1], c=colors[i], marker='x', label=f'Client {i} Samples')

    # Plot prototype variance as ellipse
    cov_128 = np.diag(prototypes[i].var)  # (128, 128)
    cov_2d = pca.components_ @ cov_128 @ pca.components_.T  # (2, 2)
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)  # 2 stddev
    ellipse = Ellipse(xy=proto_2d[0], width=width, height=height, angle=theta,
                      edgecolor='red', fc='None', lw=2, label='Prototype Var (2σ)')
    plt.gca().add_patch(ellipse)
    
    # Plot wasserstein distance lines to global prototype
    plt.plot([proto_2d[0, 0], proto_global_2d[0, 0]], [proto_2d[0, 1], proto_global_2d[0, 1]], 'k--', label='Distance to Global Prototype')
    plt.text((proto_2d[0, 0] + proto_global_2d[0, 0]) / 2, (proto_2d[0, 1] + proto_global_2d[0, 1]) / 2, f"{wasserstein_distances[i]:.2f}", color='black')



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


# print("Plot Wasserstein distances between client prototypes:")
# for i, dist in enumerate(wasserstein_distances):
#     print(f"Distance between Global Prototype and Client {i} Prototype: {dist:.4f}")
#     proto
    

# # PLot line between prototype means and write distances
# proto_c2_2d = pca.transform(prototypes[2].mean.reshape(1, -1))
# proto_c1_2d = pca.transform(prototypes[1].mean.reshape(1, -1))
# proto_c0_2d = pca.transform(prototypes[0].mean.reshape(1, -1))
# plt.plot([proto_c0_2d[0, 0], proto_c1_2d[0, 0]], [proto_c0_2d[0, 1], proto_c1_2d[0, 1]], 'k--')
# midpoint_c0_c1 = [(proto_c0_2d[0, 0] + proto_c1_2d[0, 0]) / 2, (proto_c0_2d[0, 1] + proto_c1_2d[0, 1]) / 2]
# plt.text(midpoint_c0_c1[0], midpoint_c0_c1[1], f"{wasserstein_distances[0]:.2f}", color='black')
# plt.plot([proto_c0_2d[0, 0], proto_c2_2d[0, 0]], [proto_c0_2d[0, 1], proto_c2_2d[0, 1]], 'k--')
# midpoint_c0_c2 = [(proto_c0_2d[0, 0] + proto_c2_2d[0, 0]) / 2, (proto_c0_2d[0, 1] + proto_c2_2d[0, 1]) / 2]
# plt.text(midpoint_c0_c2[0], midpoint_c0_c2[1], f"{wasserstein_distances[1]:.2f}", color='black')
# plt.plot([proto_c1_2d[0, 0], proto_c2_2d[0, 0]], [proto_c1_2d[0, 1], proto_c2_2d[0, 1]], 'k--')
# midpoint_c1_c2 = [(proto_c1_2d[0, 0] + proto_c2_2d[0, 0]) / 2, (proto_c1_2d[0, 1] + proto_c2_2d[0, 1]) / 2]
# plt.text(midpoint_c1_c2[0], midpoint_c1_c2[1], f"{wasserstein_distances[2]:.2f}", color='black')


plt.title(f"Data and Prototype Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))
plt.grid()
plt.show()
