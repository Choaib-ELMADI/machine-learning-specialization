import numpy as np

#! NumPy routines which allocate memory and fill arrays with value

a = np.zeros(4)
print(f"np.zeros(4)    => a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.zeros((4,))
print(f"np.zeros((4,)) => a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4)
print(
    f"np.random.random_sample(4) => a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n"
)

#! NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument

a = np.arange(4.0)
print(f"np.arange(4.)     => a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.rand(4)
print(f"np.random.rand(4) => a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")

#! NumPy routines which allocate memory and fill with user specified values

a = np.array([5, 4, 3, 2])
print(
    f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}"
)

a = np.array([5.0, 4, 3, 2])
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
