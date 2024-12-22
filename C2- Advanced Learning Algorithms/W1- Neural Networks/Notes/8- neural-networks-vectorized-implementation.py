import numpy as np


def g(Z):
    """
    Computes the sigmoid function
    """

    return 1 / (1 + np.exp(-Z))


def dense(A_in, W, B):
    """
    Defines a dense layer in a neural network, with a vectorized implementation
    """

    Z = np.matmul(A_in, W) + B
    # Z = A_in @ W + B              @ === matmul
    A_out = g(Z)
    return A_out


def sequential(X, W1, B1, W2, B2, W3, B3, W4, B4):
    """
    Defines a linear stack of layers for a neural network, with a vectorized implementation
    """

    A0 = X

    A1 = dense(A0, W1, B1)
    A2 = dense(A1, W2, B2)
    A3 = dense(A2, W3, B3)
    A4 = dense(A3, W4, B4)

    f_x = A4
    return f_x
