# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# 1. Create a simple Titanic dataset
# -----------------------------
data = {
    "Age": [22, 38, 26, 35, 28, 40, 50, 18, 30, 45, 60, 25, 32, 36, 48],
    "Fare": [7.25, 71.83, 7.92, 53.1, 8.05, 27.9, 30.0, 6.75, 10.5, 80.0, 20.0, 15.0, 40.0, 55.0, 100.0],
    "Survived": [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["Age", "Fare"]]   # Features
y = df["Survived"]        # Target

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------


# -----------------------------
# 4. Random Forest Classifier
# -----------------------------
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

y_pred_forest = forest_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_forest))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_forest))

