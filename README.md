# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare: One-hot encode categorical features and align target columns.
2. Scale: Standardize feature ($X$$X$) and target ($y$$y$) matrices to stabilize training.
3. Initialize: Append a column of ones to $X$$X$ for the bias term and set weights ($\theta$$\theta$) to zero.
4. Optimize: Iteratively update weights using the formula $\theta \leftarrow \theta - \alpha \nabla J(\theta)$$\theta \leftarrow \theta - \alpha \nabla J(\theta)$ until the loss converges.
5. Predict: Scale new query samples, compute predictions ($X_{new}\theta$$X_{new}\theta$), and inverse-transform back to original target units.

## Program:
```python
Program to implement the linear regression using gradient descent.
Developed by: Naresh v
RegisterNumber:  212224060169
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Create a mock CSV for demonstration since we are in a clean environment
import io
mock_data = """R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
"""

try:
    data = pd.read_csv("exp_3_50_Startups.csv")
except FileNotFoundError:
    print("exp_3_50_Startups.csv not found. Loading local mock dataset.")
    data = pd.read_csv(io.StringIO(mock_data))

print("First 5 rows of the dataset:")
display(data.head())

# One-hot encode the 'State' column
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)
print("Encoded dataset:")
display(data_encoded.head())

# Independent features (all columns except Profit)
X_raw = data_encoded.drop('Profit', axis=1).values
# Target variable (Profit)
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

# Add a column of ones to X for bias term
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)

def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost

def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history

alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())

plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()


```

## Output:

<img width="972" height="605" alt="Screenshot 2026-10-28 091806" src="https://github.com/user-attachments/assets/a7d1e1e7-25fb-4933-924c-55ff7eabc302" />

<img width="645" height="382" alt="Screenshot 2026-10-28 091844" src="https://github.com/user-attachments/assets/6ce7fcdc-4b95-40fe-9caf-f2f964771250" />

<img width="822" height="497" alt="Screenshot 2026-10-28 091855" src="https://github.com/user-attachments/assets/fb6bd290-0061-4196-82e5-0e37f1c5376d" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
