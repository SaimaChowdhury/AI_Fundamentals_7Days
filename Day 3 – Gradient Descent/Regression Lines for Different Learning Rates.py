import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a simple dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5])  # Features
y = np.array([2, 4, 6, 8, 10])  # Target (perfect linear relation: y = 2x)

X = X.reshape(-1, 1)


# -----------------------------
# 2. Gradient Descent function
# -----------------------------
def gradient_descent(X, y, alpha, epochs):
    m, b = 0, 0  # initialize slope and intercept
    for i in range(epochs):
        y_pred = m * X + b
        error = y - y_pred

        # Gradients
        dm = -2 * np.mean(X * error)
        db = -2 * np.mean(error)

        # Update parameters
        m -= alpha * dm
        b -= alpha * db

    return m, b


# -----------------------------
# 3. Experiment with learning rates
# -----------------------------
learning_rates = [0.001, 0.01, 0.1]
epochs = 200

plt.figure(figsize=(8, 6))

# Plot original data
plt.scatter(X, y, color="black", label="Data points")

# Train and plot regression line for each learning rate
colors = ["red", "green", "blue"]

for alpha, color in zip(learning_rates, colors):
    m, b = gradient_descent(X, y, alpha, epochs)
    y_pred = m * X + b
    plt.plot(X, y_pred, color=color, label=f"LR={alpha}, m={m:.2f}, b={b:.2f}")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regression Lines for Different Learning Rates")
plt.legend()
plt.show()
