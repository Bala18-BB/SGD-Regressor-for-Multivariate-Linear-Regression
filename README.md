# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Get the independent variable X and dependent variable Y.
2.Calculate the mean of the X -values and the mean of the Y -values.
3.Find the slope m of the line of best fit using the formula. 

## Program:
```
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

/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Balanagaraj.G
RegisterNumber:  25010215
*/
```

## Output:
<img width="776" height="25" alt="Screenshot 2026-02-26 100913" src="https://github.com/user-attachments/assets/2ec98973-75cf-4bed-ab80-28d418b21704" />
<img width="824" height="37" alt="Screenshot 2026-02-26 100905" src="https://github.com/user-attachments/assets/e2338013-144b-4d6a-a8df-0fc1ce2bc52f" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
