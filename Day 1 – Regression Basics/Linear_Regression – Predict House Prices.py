# Import libraries
#----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example dataset: House size (sq ft) vs Price ($1000s)
X = np.array([500, 800, 1000, 1200, 1500, 1800]).reshape(-1, 1)  # Features
y = np.array([150, 200, 250, 280, 320, 360])  # Target

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict price for a 1300 sq ft house
predicted_price = model.predict([[1300]])
print("Predicted price for 1300 sq ft house:", predicted_price[0])

# Plot results
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.legend()
plt.show()
