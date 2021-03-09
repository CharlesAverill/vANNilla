from .random import Random


def scalar_dot(m, n):
    """
    :param m: Vector
    :param n: Vector
    :return: Scalar product of m and n
    """
    return sum(m_i * n_i for m_i, n_i in zip(m, n))


def identity(n):
    """
    :param n: Output length, width
    :return: Square identity matrix of shape (n, n)
    """
    return [([0] * i) + [1] + ([0] * (n - i - 1)) for i in range(n)]


def zeros(dims):
    """
    :param dims: Shape to fill
    :return: Matrix of shape dims filled with 0.0
    """
    if type(dims) == int:
        return 0
    if len(dims) == 1:
        return [0.0] * dims[0]
    return [zeros(dims[1:]) for _ in range(dims[0])]


def list_prod(ls):
    """
    :param ls: A one-dimensional list
    :return: The product of all values in ls
    """
    prod = 1
    for dim in ls:
        prod *= dim
    return prod


def from_iterator(iterator, dims):
    """
    :param iterator: Iterator to pull values from
    :param dims: Shape to fill
    :return: Matrix with shape dims, filled with
             values from iterator
    """
    if dims == ():
        return [next(iterator)]
    if type(dims) == int:
        return next(iterator)
    if len(dims) == 1:
        out = []
        for i in range(dims[0]):
            out.append(next(iterator))
        return out
    return [from_iterator(iterator, dims[1:]) for _ in range(dims[0])]


def random(dims, min_val=0, max_val=1, rng=None):
    """
    :param dims: Shape to fill
    :param min_val: Minimum random value
    :param max_val: Maximum random value
    :param rng: Random Number Generator, built recursively
    :return: Matrix with shape dims filled with random
             values between [min_val, max_val)
    """
    if not rng:
        rng = Random()
    if len(dims) == 1:
        return [rng.next(min_val, max_val)] * dims[0]
    return [random(dims[1:], min_val, max_val, rng) for _ in range(dims[0])]


def multiply_accumulate(m, index, a=1, b=0):
    """
    :param m: Matrix to take MAC of
    :param index: Output of multidim_enumerate(m)
    :param a: Multiplier
    :param b: Constant
    """
    for i in index[:-1]:
        m = m[i]
    m[index[-1]] = a * m[index[-1]] + b
