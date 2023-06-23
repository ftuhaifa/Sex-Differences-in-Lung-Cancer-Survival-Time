import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Load the data from CSV
data = pd.read_csv("LungCancer25.csv")

# Clean up column names by removing leading/trailing whitespace
data.columns = data.columns.str.strip()

# Create a new DataFrame with the necessary columns
df = data[["Survival years", "Sex", "Surgery status"]].copy()

df = df[df["Surgery status"] != 3]

# Convert categorical variables to one-hot encoded vectors
X_encoded = pd.get_dummies(df.drop("Survival years", axis=1), drop_first=True)
y = df["Survival years"].values

# Concatenate the encoded features with the numerical features
X = np.concatenate([X_encoded.values, df[["Survival years"]].values], axis=1)

# Normalize the input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM (samples, timesteps, features)
n_features = X_train.shape[1]
n_timesteps = 1  # Since we are not using sequential information in this example
X_train = np.reshape(X_train, (X_train.shape[0], n_timesteps, n_features))
X_test = np.reshape(X_test, (X_test.shape[0], n_timesteps, n_features))

# Build the LSTM model
def create_model(units):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Define the parameter grid for grid search
param_grid = {
    'units': [50, 100],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

# Create the model
model = KerasRegressor(build_fn=create_model, verbose=0)

# Create the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Convert the predicted values to integers
#y_pred = np.round(y_pred).astype(int)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Plot the predicted and actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Survival years')
plt.legend
