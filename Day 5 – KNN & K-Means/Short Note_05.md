**📘 Day 6 – KNN & K‑Means**

🎯 K‑Nearest Neighbors (KNN)
- Definition: KNN is a classification algorithm that predicts a label based on the majority vote of its k nearest neighbors.
- Analogy: Imagine moving into a new neighborhood. To guess your favorite food, people look at your closest neighbors — if most of them love pizza, they assume you do too.
Key Ideas
- Distance Metric: Usually Euclidean distance (straight‑line distance).
- Choice of k:
- Small k → sensitive to noise.
- Large k → smoother but may ignore local details.
- Lazy Learner: KNN doesn’t build a model; it just stores data and compares at prediction time.

🎯 K‑Means Clustering
- Definition: K‑Means is an unsupervised algorithm that groups data into k clusters based on similarity.
- Analogy: Imagine sorting marbles by color. K‑Means tries to place each marble into one of k buckets so that marbles in the same bucket are similar.
Key Ideas
- Centroids: Each cluster has a center point (mean).
- Iterations: Assign points to nearest centroid → update centroids → repeat until stable.
- Unsupervised: No labels required; it discovers structure in data.

📊 Metrics
- KNN: Accuracy, Precision, Recall.
- K‑Means: Inertia (sum of squared distances), Silhouette Score.

🧪 Practice Ideas
- Train a KNN classifier on a labeled dataset (e.g., Iris flowers).
- Experiment with different values of k and distance metrics.
- Apply K‑Means clustering to unlabeled data (e.g., customer segmentation).
- Visualize clusters with scatter plots.

📝 Reflection
- KNN feels like asking your neighbors for advice — simple but effective.
- K‑Means is like organizing things into groups without knowing labels beforehand.
- Both algorithms rely on distance and similarity as their main idea.
- Visualizing clusters and neighbors makes the concepts much easier to grasp

