import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# -----------------------------
# 1. Create synthetic dataset
# -----------------------------
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# -----------------------------
# 2. Apply K-Means clustering (choose k=4)
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# -----------------------------
# 3. Visualize clusters
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap="viridis", s=30)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means Clustering with k=4")
plt.legend()
plt.show()