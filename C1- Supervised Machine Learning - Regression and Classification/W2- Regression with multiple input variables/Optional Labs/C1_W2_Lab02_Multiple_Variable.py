import copy, math
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("./deeplearning.mplstyle")
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


def predict_single_loop(x, w, b):
    """
    Single predict using linear regression

    Args:
        x (ndarray) : shape (n,) example with multiple features
        w (ndarray) : shape (n,) model parameters
        b (scalar)  :            model parameter

    Returns:
        p (scalar)  : prediction
    """

    n = x.shape[0]
    p = 0

    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i

    p = p + b
    return p


def predict(x, w, b):
    """
    Single predict using linear regression

    Args:
        x (ndarray) : shape (n,) example with multiple features
        w (ndarray) : shape (n,) model parameters
        b (scalar)  :            model parameter

    Returns:
        p (scalar)  : prediction
    """

    p = np.dot(x, w) + b
    return p


def compute_cost(X, y, w, b):
    """
    Computes cost

    Args:
        X (ndarray (m,n)) : data, m examples with n features
        y (ndarray (m,))  : target values
        w (ndarray (n,))  : model parameters
        b (scalar)        : model parameter

    Returns:
        cost (scalar)     : computed cost
    """

    m = X.shape[0]  # Number of examples
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
        X (ndarray (m,n)) : data, m examples with n features
        y (ndarray (m,))  : target values
        w (ndarray (n,))  : model parameters
        b (scalar)        : model parameter

    Returns:
        dj_dw (ndarray (n,)) : the gradient of the cost with respect to the parameters w
        dj_db (scalar)       : the gradient of the cost with respect to the parameter b
    """

    m, n = X.shape  # number of examples, number of features
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b.
    Updates w and b by taking num_iters gradient steps with learning rate alpha.

    Args:
        X         (ndarray (m,n)) : data, m examples with n features
        y         (ndarray (m,))  : target values
        w_in      (ndarray (n,))  : initial model parameters
        b_in      (scalar)        : initial model parameter
        alpha     (float)         : learning rate
        num_iters (int)           : number of iterations to run gradient descent

    Returns:
        w         (ndarray (n,))  : updated values of parameters w
        b         (scalar)        : updated value of parameter b
        J_history                 : for graphing purposes
    """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        #! Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        #! Update parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        # if i % math.ceil(num_iters / 10) == 0:
        # print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
# print(f"X Shape: {X_train.shape}, X Type: {type(X_train)}")
# print(X_train)
# print(f"y Shape: {y_train.shape}, y Type: {type(y_train)}")
# print(y_train)

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
# print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

x_vec = X_train[0, :]
# print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
f_wb = predict_single_loop(x_vec, w_init, b_init)
# print(f"LOOP: f_wb shape {f_wb.shape}, prediction: {f_wb}")  # type: ignore
f_wb = predict(x_vec, w_init, b_init)
# print(f".DOT: f_wb shape {f_wb.shape}, prediction: {f_wb}")  # type: ignore

cost = compute_cost(X_train, y_train, w_init, b_init)
# print(f"Cost at optimal w: {cost}")

tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)
# print(f"dj_dw at initial w,b: \n {tmp_dj_dw}")
# print(f"dj_db at initial w,b: {tmp_dj_db}")

initial_w = np.zeros_like(w_init)
initial_b = 0.0
iterations = 1_000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(
    X_train, y_train, initial_w, initial_b, alpha, iterations
)
print(f"b,w found by gradient descent: {b_final:0.2f}, {w_final} ")
m, _ = X_train.shape
for i in range(m):
    print(
        f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}"
    )

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

ax1.plot(J_hist)
ax1.set_title("Cost vs. iteration")
ax1.set_xlabel("iteration step")
ax1.set_ylabel("Cost")

ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Cost vs. iteration (tail)")
ax2.set_xlabel("iteration step")
ax2.set_ylabel("Cost")

plt.show()
