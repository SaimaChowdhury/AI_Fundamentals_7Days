import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create a simple dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5])  # Features
y = np.array([2, 4, 6, 8, 10])  # Target (perfect linear relation: y = 2x)

# Reshape X for matrix operations
X = X.reshape(-1, 1)

# -----------------------------
# 2. Initialize parameters
# -----------------------------
m = 0  # slope
b = 0  # intercept
alpha = 0.01  # learning rate
epochs = 1000  # number of iterations

errors = []

# -----------------------------
# 3. Gradient Descent loop
# -----------------------------
for i in range(epochs):
    y_pred = m * X + b
    error = y - y_pred

    # Mean Squared Error
    mse = np.mean(error ** 2)
    errors.append(mse)

    # Gradients
    dm = -2 * np.mean(X * error)
    db = -2 * np.mean(error)

    # Update parameters
    m -= alpha * dm
    b -= alpha * db

print("Final slope (m):", m)
print("Final intercept (b):", b)

# -----------------------------
# 4. Plot regression line
# -----------------------------
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, m * X + b, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# -----------------------------
# 5. Plot error curve
# -----------------------------
plt.plot(range(epochs), errors, color="green")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("Error Curve during Gradient Descent")
plt.show()