import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

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


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for a linear regression.

    Args:
        x    (ndarray (m,)) : data, m training examples
        y    (ndarray (m,)) : target values
        w, b (scalar)       : model parameters

    Returns:
        dj_dw (scalar)      : the gradient of the cost with respect to the parameters w
        dj_db (scalar)      : the gradient of the cost with respect to the parameter b
    """

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w, b.
    Updates w, b by taking num_iters gradient steps with learning rate alpha.

    Args:
        x          (ndarray (m,)) : data, m examples
        y          (ndarray (m,)) : target values
        w_in, b_in (scalar)       : initial values of model parameters
        alpha      (float)        : learning rate
        num_iters  (int)          : number of iterations to run gradient descent

    Returns:
        w         (scalar) : updated value of parameter w after running gradient descent
        b         (scalar) : updated value of parameter b after running gradient descent
        J_history (List)   : history of cost values
        p_history (list)   : history of parameters [w,b]
    """

    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        #! Calculate the gradient
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        #! Update parameters
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration (only first 10_0000 iterations)
        if i < 10_0000:
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        # if i % math.ceil(num_iters / 10) == 0:
        #     print(
        #         f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
        #         f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
        #         f"w: {w: 0.3e}, b: {b: 0.5e}",
        #     )

    return w, b, J_history, p_history


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w_init = 0
b_init = 0
tmp_alpha = 1.0e-2
iterations = 10_000

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations
)
print(f"(w, b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel("Cost")
ax2.set_ylabel("Cost")
ax1.set_xlabel("iteration step")
ax2.set_xlabel("iteration step")
plt.show()
