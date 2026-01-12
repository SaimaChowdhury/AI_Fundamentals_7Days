import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# -----------------------------
# 1. Create synthetic dataset
# -----------------------------
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# -----------------------------
# 2. Compare K-Means with different k values
# -----------------------------
plt.figure(figsize=(12, 10))

# Try k = 2, 3, 4, 5
for i, k in enumerate([2, 3, 4, 5], start=1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plot each result in a different slot
    plt.subplot(2, 2, i)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", s=30)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="red", marker="X", s=200, label="Centroids")
    plt.title(f"K-Means with k={k}")
    plt.legend()

plt.tight_layout()
plt.show()