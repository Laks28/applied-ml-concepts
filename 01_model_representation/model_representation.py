"""
01- Model Representation
Goal: Implement a simple linear model to predict house prices
based on size using f_wb = w * x + b
"""

import numpy as np
import matplotlib.pyplot as plt

# Training data 
x_train = np.array([1.0, 2.0]) # features (size in 1000 sq.ft)
y_train = np.array([300.0, 500.0]) # target values (price in $1000)

print(f"x_train={x_train}")
print(f"y_train={y_train}")

# Check 
i =0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i}) = ({x_i}), ({y_i})")

# Get # of training examples 
print(f"x_train.shape = {x_train.shape}")
m= x_train.shape[0]
print(f"Number of training examples: {m}")

# Plot training data
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.xlabel("Size (in 1000 sq ft)")
plt.ylabel("Prices( in 1000s of dollars)")
plt.grid(True)
plt.show()

#Define Model Function
def compute_model_output(x, w, b):
 """
Computes prediction of linear model f_wb = w * x + b
 """

 m = x.shape[0]
 f_wb = np.zeros(m)
 for i in range (m):
    f_wb[i] = w * x[i] + b
 return f_wb

# Test with parameters 
w = 200
b = 100

# Compute model predictions
tmp_f_wb = compute_model_output(x_train,w,b)

print("\nPredictions")
for i in range(m):
    print(f"f_wb(x^{i}) = {tmp_f_wb[i]:.2f} for x^{i} = {x_train[i]}")

# Plot predicted vs actual data
plt.plot(x_train, tmp_f_wb, c='b', label='Prediction line (Our model)')
plt.scatter(x_train, y_train, marker='x', c='r', label='Training data')
plt.title("Housing Prices - Model Representation")
plt.xlabel("Size (in 1000 sq ft)")
plt.ylabel("Price (in 1000s of dollars)")
plt.legend()
plt.grid(True)
plt.show()





