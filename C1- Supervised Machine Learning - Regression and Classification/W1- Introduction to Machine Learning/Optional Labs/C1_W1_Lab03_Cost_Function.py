import numpy as np

# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use("./deeplearning.mplstyle")


def compute_cost(x, y, w, b):
    """
    Computes the cost function for a linear regression model.

    Args:
        x    (ndarray (m,)) : data, m training examples
        y    (ndarray (m,)) : target values
        w, b (scalar)       : model parameters

    Returns:
        total_cost (float)  : cost of using w, b as the parameters for linear regression to fit the data points in x and y
    """

    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
plt_intuition(x_train, y_train)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array(
    [
        250,
        300,
        480,
        430,
        630,
        730,
    ]
)
plt.close("all")
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()
