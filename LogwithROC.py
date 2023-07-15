import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("LungCancer25.csv")

# Remove rows with Surgery status equal to 3
data = data[data["Surgery status"] != 3]

# Encode the target variable into binary values
data["Sex_Encoded"] = pd.get_dummies(data["Sex"], drop_first=True)

# Select the relevant columns for prediction
#X = data[["Survival years", "Surgery status"]]
X = data[["Survival years", "Surgery status", "Age","Race"]]
y = data["Sex_Encoded"]

# Balance the classes using over-sampling
#smote = SMOTE(random_state=42)
#X, y = smote.fit_resample(X, y)

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set up the parameter grid for grid search
param_grid = {
    "C": [0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
}

# Perform grid search to find the best parameters
logit_model = LogisticRegression(solver="liblinear")
logit_grid = GridSearchCV(logit_model, param_grid, cv=5)
logit_grid.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = logit_grid.best_params_
print("Best Parameters:", best_params)

# Fit the logistic regression model with the best parameters on the training set
logit_model_best = LogisticRegression(solver="liblinear", **best_params)
logit_model_best.fit(X_train, y_train)

# Predict the target variable on the testing set
y_pred = logit_model_best.predict(X_test)
y_pred_proba = logit_model_best.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
confusion = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate true positives, false positives, true negatives, false negatives
tn, fp, fn, tp = confusion.ravel()

# Calculate specificity
specificity = tn / (tn + fp)

# Print evaluation metrics and confusion matrix
print("Accuracy:", accuracy)
print("AUC:", auc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("Confusion Matrix:")
print(confusion)
print("True Positives:", tp)
print("False Positives:", fp)
print("True Negatives:", tn)
print("False Negatives:", fn)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
