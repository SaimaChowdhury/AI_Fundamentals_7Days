import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Create dataset
# -----------------------------
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# -----------------------------
# 2. Function to plot decision boundary
# -----------------------------
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 200),
        np.linspace(ylim[0], ylim[1], 200)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
    plt.title(title)
    plt.show()


# -----------------------------
# 3. Compare activation functions
# -----------------------------
activations = ["logistic", "tanh", "relu"]

for act in activations:
    nn = MLPClassifier(hidden_layer_sizes=(5,), activation=act, max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    print(f"Activation={act}, Accuracy={accuracy_score(y_test, y_pred):.2f}")
    plot_decision_boundary(nn, X, y, f"NN with {act} activation")