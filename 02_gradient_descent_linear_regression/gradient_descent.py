"""
Gradient Descent for Linear Regression

Goal: Find w and b that best fit the data by reducing cost.

Dataset:
Size (1000 sqft): [1.0, 2.0]
Price ($1000s):   [300.0, 500.0]
"""

import numpy as np
import matplotlib.pyplot as plt

# Load dataset
x_train = np.array([1.0, 2.0])        # feature: size in 1000 sqft
y_train = np.array([300.0, 500.0])    # target: price in $1000s


# Calculate Cost (Mean Squared Error)
def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    Mean Squared Error:
    J(w,b) = (1/2m) * Σ ( (w*x + b - y)^2 )
    """
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = w * x[i] + b          # prediction is w*x + b 
        cost += (f_wb - y[i]) ** 2

    return cost / (2 * m)

# Compute Gradients
def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    """
    Gradients:
      dj_dw = (1/m) * Σ ( (w*x + b - y) * x )
      dj_db = (1/m) * Σ ( (w*x + b - y) )
    """
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0

    for i in range(m):
        f_wb = w * x[i] + b
        err = f_wb - y[i]
        dj_dw += err * x[i]         # accumulate with +=
        dj_db += err                # accumulate with +=

    return dj_dw / m, dj_db / m


# Gradient Descent
def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w_init: float,
    b_init: float,
    alpha: float,
    num_iters: int,
) -> tuple[float, float, list[float]]:
    """
    Runs gradient descent and returns: final w, final b, and cost history.
    """
    w, b = w_init, b_init
    J_history: list[float] = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # subtract learning-rate * gradient
        w -= alpha * dj_dw
        b -= alpha * dj_db

        J_history.append(compute_cost(x, y, w, b))

        # Print progress roughly 10 times during run
        if num_iters >= 10 and i % max(1, num_iters // 10) == 0:
            print(
                f"Iteration {i:4d}: Cost {float(J_history[-1]):.2e}  "
                f"dj_dw {dj_dw:.3e}, dj_db {dj_db:.3e}  w {w:.3f}, b {b:.3f}"
            )

    return w, b, J_history


# Plot cost vs iterations
def plot_cost(J_history: list[float]) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(J_history)
    plt.title("Cost vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_data_and_fit(x: np.ndarray, y: np.ndarray, w: float, b: float) -> None:
    """Scatter the training data and draw the fitted line y = w*x + b"""
    # generate x range for a smooth line
    x_line = np.linspace(float(x.min()) * 0.8, float(x.max()) * 1.2, 100)
    y_line = w * x_line + b

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=60, label="data")        # training points
    plt.plot(x_line, y_line, label=f"fit: y={w:.1f}x + {b:.1f}")  # fitted line
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price ($1000s)")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def high_lr_experiment(
    x: np.ndarray,
    y: np.ndarray,
    w0: float = 0.0,
    b0: float = 0.0,
    alpha: float = 0.8,
    num_iters: int = 10,
) -> None:
    """
    Runs gradient descent with a large learning rate to illustrate divergence.
    Plots the cost curve and prints the costs.
    """
    w_bad, b_bad, J_bad = gradient_descent(x, y, w0, b0, alpha, num_iters)

    plt.figure(figsize=(8, 4))
    plt.plot(J_bad, marker="o")
    plt.title(f"Cost vs Iterations (alpha = {alpha})")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("High-α costs:", [f"{c:.2e}" for c in J_bad])
    print(f"(w, b) after {num_iters} iters with alpha={alpha}: ({w_bad:.2f}, {b_bad:.2f})")



# Main
if __name__ == "__main__":
    w0, b0 = 0.0, 0.0
    alpha = 1e-2
    iterations = 10_000

    # Train
    w_final, b_final, J_hist = gradient_descent(
        x_train, y_train, w0, b0, alpha, iterations
    )

    print(f"\n(w, b) found by gradient descent: ({w_final:.4f}, {b_final:.4f})")

    def predict(x_val: float) -> float:
        return w_final * x_val + b_final

    print(f"1000 sqft house prediction: {predict(1.0):.1f} Thousand dollars")
    print(f"1200 sqft house prediction: {predict(1.2):.1f} Thousand dollars")
    print(f"2000 sqft house prediction: {predict(2.0):.1f} Thousand dollars")

    plot_data_and_fit(x_train, y_train, w_final, b_final)

    plot_cost(J_hist)

    high_lr_experiment(x_train, y_train, alpha=0.8, num_iters=10)







