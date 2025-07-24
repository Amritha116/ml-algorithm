import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs

# Create synthetic dataset
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['True cluster'] = y
print("First 5 rows of the synthetic dataset (all columns):")
print(df.head())

# Visualize original data
plt.figure(0)
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Generated Data")
plt.show()

# K-means initialization
k = 3
clusters = {}
np.random.seed(23)
for idx in range(k):
    center = 2 * (2 * np.random.random((X.shape[1],)))
    cluster = {
        'center': center,
        'points': []
    }
    clusters[idx] = cluster

# Plot initial centers
plt.scatter(X[:, 0], X[:, 1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', s=200, c='red', label=f"Initial center {i}")
plt.title("Initial Random Cluster Centers")
plt.legend()
plt.show()

# Distance function
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))
# Assign points to the nearest cluster
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        curr_x = X[idx]
        dists = [distance(curr_x, clusters[i]['center']) for i in range(k)]
        cluster_idx = np.argmin(dists)
        clusters[cluster_idx]['points'].append(curr_x)
    return clusters

# Update cluster centers
def update_clusters(clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
        clusters[i]['points'] = []
    return clusters

# Predict clusters for each point
def pred_cluster(X, clusters):
    preds = []
    for i in range(X.shape[0]):
        dists = [distance(X[i], clusters[j]['center']) for j in range(k)]
        preds.append(np.argmin(dists))
    return preds

# K-means iteration
n_iters = 10
for i in range(n_iters):
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(clusters)

# Get final cluster predictions
pred = pred_cluster(X, clusters)

# Plot final clustered data
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='viridis')
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', s=200, c='red', label=f"Final center {i}")
plt.title("Final Clustered Data (K-Means)")
plt.grid(True)
plt.legend()
plt.show()
