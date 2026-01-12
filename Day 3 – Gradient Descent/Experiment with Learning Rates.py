import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a simple dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5])  # Features
y = np.array([2, 4, 6, 8, 10])  # Target (perfect linear relation: y = 2x)

X = X.reshape(-1, 1)


# -----------------------------
# 2. Function for Gradient Descent
# -----------------------------
def gradient_descent(X, y, alpha, epochs):
    m, b = 0, 0  # initialize slope and intercept
    errors = []

    for i in range(epochs):
        y_pred = m * X + b
        error = y - y_pred
        mse = np.mean(error ** 2)
        errors.append(mse)

        # Gradients
        dm = -2 * np.mean(X * error)
        db = -2 * np.mean(error)

        # Update parameters
        m -= alpha * dm
        b -= alpha * db

    return m, b, errors


# -----------------------------
# 3. Experiment with different learning rates
# -----------------------------
learning_rates = [0.001, 0.01, 0.1]
epochs = 200

plt.figure(figsize=(8, 6))

for alpha in learning_rates:
    m, b, errors = gradient_descent(X, y, alpha, epochs)
    plt.plot(range(epochs), errors, label=f"LR={alpha}")
    print(f"Learning Rate={alpha}, Final slope={m:.2f}, intercept={b:.2f}, Final MSE={errors[-1]:.4f}")

plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("Effect of Learning Rate on Convergence")
plt.legend()
plt.show()
