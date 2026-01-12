import numpy as np
import pandas as pd
from math import log2


# -----------------------------
# 1. Define Entropy function
# -----------------------------
def entropy(class_counts):
    total = sum(class_counts)
    return -sum((count / total) * log2(count / total) for count in class_counts if count != 0)


# -----------------------------
# 2. Example dataset (toy)
# -----------------------------
# Imagine a dataset of "Play Tennis" with features like Weather
data = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny",
                "Overcast", "Overcast", "Rain"],
    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
})

# -----------------------------
# 3. Calculate Entropy of target
# -----------------------------
target_counts = data["Play"].value_counts().tolist()
print("Entropy of Play:", entropy(target_counts))


# -----------------------------
# 4. Information Gain for splitting on Weather
# -----------------------------
def info_gain(data, feature, target="Play"):
    # Entropy before split
    total_entropy = entropy(data[target].value_counts().tolist())

    # Weighted entropy after split
    values = data[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target].value_counts().tolist())

    return total_entropy - weighted_entropy


print("Information Gain (Weather):", info_gain(data, "Weather"))