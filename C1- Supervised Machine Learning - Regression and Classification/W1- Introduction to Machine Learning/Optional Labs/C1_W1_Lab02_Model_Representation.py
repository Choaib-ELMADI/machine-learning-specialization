import numpy as np
import matplotlib.pyplot as plt


def linear_regression_model(x, w, b):
    """
    Computes the predictions of a linear regression model.

    Args:
        x    (ndarray (m,)) : data, m training examples
        w, b (scalar)       : model parameters

    Returns:
        f_wb (ndarray (m,)) : model predictions
    """

    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
x_train_shape = x_train.shape  # shape = (rows, columns)
m = x_train.shape[0]

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")
print(f"x_train.shape: {x_train_shape}")
print(f"Number of training examples is: {m}")

w = 200
b = 100
predicted_values = linear_regression_model(x_train, w, b)

plt.scatter(x_train, y_train, marker="x", c="r", label="Reel Values")
plt.scatter(x_train, predicted_values, marker="x", c="b", label="Predicted Values")
plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (in 1000s of dollars)")
plt.legend()
plt.show()
