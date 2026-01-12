import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# -----------------------------
# 1. Create dataset
# -----------------------------
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 2. Train NN and visualize loss curve
# -----------------------------
nn = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", max_iter=1000, random_state=42)
nn.fit(X_train, y_train)

plt.plot(nn.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve during Training")
plt.show()