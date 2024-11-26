import numpy as np
import time


def my_dot(a, b):
    """
    Computes the dot product of two vectors.

    Args:
        a (ndarray (n,)) : input vector
        b (ndarray (n,)) : input vector with same dimension as a

    Returns:
        x (scalar)
    """

    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


# ===>
a = np.zeros(4)
a = np.zeros((4,))
a = np.random.random_sample(4)
a = np.arange(4.0)
a = np.random.rand(4)
a = np.array([5, 4, 3, 2])
a = np.array([5.0, 4, 3, 2])

# ===>
b = np.arange(10)

# ===>
c = np.arange(10)
d = c[2:7:1]
d = c[2:7:2]
d = c[3:]
d = c[:3]
d = c[:]

# ===>
e = np.array([1, 2, 3, 4])
f = -e
f = np.sum(e)
f = np.mean(e)
f = e**2

# ===>
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
c = a + b

# ===>
a = np.array([1, 2, 3, 4])
b = 5 * a

# ===>
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = my_dot(a, b)
d = np.dot(a, b)

# ===>
np.random.seed(1)
a = np.random.rand(10_000_000)  # VERY LARGE ARRAYS
b = np.random.rand(10_000_000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

tic = time.time()  # capture start time
c = my_dot(a, b)
toc = time.time()  # capture end time

del a
del b  # REMOVE THESE BIG ARRAYS FROM MEMORY

# ===>
X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)

# ===>
a = np.zeros((1, 5))
a = np.zeros((2, 1))
a = np.random.random_sample((1, 1))

# ===>
a = np.arange(6).reshape(-1, 2)  # reshape is a convenient way to create matrices

# ===>
a = np.arange(20).reshape(-1, 10)
b = a[0, 2:7:1]  # 1-D array
c = a[:, 2:7:1]  # 2-D array
d = a[:, :]
e = a[1, :]
f = a[1]
