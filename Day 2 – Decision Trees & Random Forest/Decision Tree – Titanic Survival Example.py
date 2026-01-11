# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset: Titanic survival (1 = survived, 0 = not survived)
data = {
    "Age": [22, 38, 26, 35, 28, 40, 50, 18, 30, 45],
    "Fare": [7.25, 71.83, 7.92, 53.1, 8.05, 27.9, 30.0, 6.75, 10.5, 80.0],
    "Survived": [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[["Age", "Fare"]]   # Features
y = df["Survived"]        # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Decision Tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# Predictions
y_pred = tree_model.predict(X_test)

# Evaluate accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(tree_model, feature_names=["Age", "Fare"], class_names=["Not Survived", "Survived"], filled=True)
plt.show()
