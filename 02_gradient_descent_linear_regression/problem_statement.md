# 02 - Gradient Descent for Linear Regression

This module is about using **Gradient Descent** to train a simple **Linear Regression** model.  
The goal is to find the best fiting parameters `w` and `b` that fit the data by reducing the prediction error.

# Problem Statement

Given a small dataset of house prices, the task is to predict prices based on house size.  
The model used is:

\[
f(x) = w \cdot x + b
\]

### Dataset

| Size (1000 sqft) | Price ($1000s) |
|------------------|----------------|
| 1                | 300            |
| 2                | 500            |


The relationship between size and price can be modeled using:

\[
f_{w,b}(x) = w \cdot x + b
\]

The idea is to adjust `w` and `b` step by step so that the predictions come closer to the actual prices.

### Objective

- How to calculate **cost** to measure how far predictions are from actual values  
- How to find **gradients** to know which direction to move `w` and `b`  
- How **gradient descent** updates parameters to lower the cost  
- Why **learning rate** matters for convergence  
- How to plot and visualize learning progress


## Key Formulas

**Cost Function:**

\[
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})^2
\]

**Gradients:**

\[
\frac{\partial J}{\partial w} = \frac{1}{m} \sum (f(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
\]
\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum (f(x^{(i)}) - y^{(i)})
\]

**Parameter Updates:**

\[
w = w - \alpha \cdot \frac{\partial J}{\partial w}
\]
\[
b = b - \alpha \cdot \frac{\partial J}{\partial b}
\]

Here, `α` is the learning rate.

---

## Steps Followed

1. Start with `w = 0` and `b = 0`  
2. Use the dataset to calculate the cost  
3. Find the gradients for `w` and `b`  
4. Update `w` and `b` using the formulas above  
5. Repeat the process until the cost becomes very small

---

## Example Output

```text
Iteration    0: Cost 7.93e+04  w: 6.50, b: 4.00
Iteration 1000: Cost 3.41e+00  w: 194.9, b: 108.2
Iteration 9000: Cost 2.90e-05  w: 199.9, b: 100.0
(w,b) found by gradient descent: (199.99, 100.01)