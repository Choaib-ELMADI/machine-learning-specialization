import numpy as np


def g(z):
    """
    Computes the sigmoid function
    """

    return 1 / (1 + np.exp(-z))


def dense(a_in, W, b):
    """
    Defines a dense layer in a neural network
    """

    units = W.shape[1]  # get the number of neurons in a layer
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in[j]) + b[j]
        a_out[j] = g(z)

    return a_out


def sequential(x, W1, b1, W2, b2, W3, b3, W4, b4):
    """
    Defines a linear stack of layers for a neural network
    """

    a0 = x

    a1 = dense(a0, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)

    f_x = a4
    return f_x
