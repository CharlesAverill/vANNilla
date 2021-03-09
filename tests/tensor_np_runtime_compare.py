import timeit

np_add = (
    "import numpy as np\n"
    "np.array([[1, 2, 3], [4, 5, 6]]) + "
    "np.array([[7, 8, 9], [10, 11, 12]])"
)
t_add = (
    "from vANNilla.utils import Tensor\n"
    "Tensor([[1, 2, 3], [4, 5, 6]]) + "
    "Tensor([[7, 8, 9], [10, 11, 12]])"
)

print("Testing addition")
add_times = [
    timeit.timeit(np_add, number=500),
    timeit.timeit(t_add, number=500),
]
print(f"numPy: {add_times[0]}")
print(f"Tensor: {add_times[1]}")
print(f"numPy {add_times[1] / add_times[0]}x faster than Tensor\n")

np_dot = (
    "import numpy as np\n"
    "np.dot(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]]).T)"
)
t_dot = (
    "from vANNilla.utils import Tensor\n"
    "Tensor([[7, 8, 9], [10, 11, 12]]).transposed * Tensor([[1, 2, 3], [4, 5, 6]])"
)

print("Testing inner product")
add_times = [
    timeit.timeit(np_dot, number=500),
    timeit.timeit(t_dot, number=500),
]
print(f"numPy: {add_times[0]}")
print(f"Tensor: {add_times[1]}")
print(f"numPy {add_times[1] / add_times[0]}x faster than Tensor\n")
