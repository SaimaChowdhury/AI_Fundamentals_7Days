import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Create a simple linearly separable dataset
# -----------------------------
X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 2. Train Linear SVM
# -----------------------------
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)

y_pred = linear_svm.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred))


# -----------------------------
# 3. Visualize decision boundary
# -----------------------------
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30)

    # Create grid
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
    plt.title(title)
    plt.show()


plot_decision_boundary(linear_svm, X, y, "Linear SVM Decision Boundary")
