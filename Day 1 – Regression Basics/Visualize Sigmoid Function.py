import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y, color="green")
plt.title("Sigmoid Function")
plt.xlabel("Input (z)")
plt.ylabel("Output (Probability)")
plt.grid()
plt.show()
