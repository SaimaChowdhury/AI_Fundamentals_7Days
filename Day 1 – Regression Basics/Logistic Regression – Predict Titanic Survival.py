# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Simple dataset: Titanic survival (1 = survived, 0 = not survived)
data = {
    "Age": [22, 38, 26, 35, 28, 40, 50],
    "Fare": [7.25, 71.83, 7.92, 53.1, 8.05, 27.9, 30.0],
    "Survived": [0, 1, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df[["Age", "Fare"]]   # Features
y = df["Survived"]        # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
