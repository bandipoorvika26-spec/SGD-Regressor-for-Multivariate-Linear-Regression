# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure. 2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets. 3.Create a linear regression model and fit it using the training data. 4.Predict the results for the testing set and plot the training and testing sets with fitted lines. 5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance. 

## Program:
#Manual Implementation using Numpy
import numpy as np

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)

# ------------------------------
# Step 2: Feature normalization
# ------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

# ------------------------------
# Step 3: Initialize weights
# ------------------------------
n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# ------------------------------
# Step 4: Stochastic Gradient Descent
# ------------------------------
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)
#Using scikit-learn SGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Features and target
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
])
y = np.array([50, 45, 70, 80, 95])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SGD Regressor
sgd_reg = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_reg.fit(X_scaled, y)

# Coefficients and intercept
print("Weights (coefficients):", sgd_reg.coef_)
print("Intercept:", sgd_reg.intercept_)

# Predictions
y_pred = sgd_reg.predict(X_scaled)
print("Predicted values:", y_pred)
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: BANDI POORVIKA
RegisterNumber:  25013262
*/
```

## Output:
Trained Weights (including intercept): [68.00600913 10.21035043  3.21184007  6.00928327]
Predicted values: [49.53368945 44.96389764 70.63122027 80.51508398 94.38615431]


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
