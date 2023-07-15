# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:13:31 2023

@author: ftuha
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("LungCancer25.csv")

# Remove rows with Surgery status equal to 3
data = data[data["Surgery status"] != 3]

# Encode the target variable into binary values
data["Sex_Encoded"] = pd.get_dummies(data["Sex"], drop_first=True)

# Select the relevant columns for prediction
X = data[["Survival years", "Surgery status", "Age", "Race"]]
y = data["Sex_Encoded"]

# Initialize and fit the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Get feature importances
importances = rf_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for i, index in enumerate(indices):
    print(f"{i+1}. Feature '{X.columns[index]}' ({importances[index]})")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
