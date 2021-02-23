from .random import Random


def scalar_dot(m, n):
    # Scalar dot product of vectors m and n
    return sum(m_i * n_i for m_i, n_i in zip(m, n))


def dot_prod(m, n):
    # Vector outer dot product of vectors m and n
    if len(m[0]) != len(n):
        raise RuntimeError("Dimension mismatch: M height should equal N width")
    out = []
    for row in m:
        out.append(sum([row[i] * n[i] for i in range(len(n))]))

    return out


def shape(m):
    # Tuple of shape of vector m, accounts for raggedness
    m_shape = []
    while True:
        try:
            iter(m)
            last_len = len(m[0])
            for row in m:
                iter(row)
                if len(row) != last_len:
                    break
                last_len = len(row)
            m_shape.append(len(m))
            m = m[0]
        except TypeError:
            if type(m) == list:
                m_shape.append(len(m))
            break
    return tuple(m_shape)


def size(m):
    # Returns scalar size of matrix m
    m_shape = shape(m)
    prod = 1
    for dim in m_shape:
        prod *= dim
    return prod


def transpose(m):
    # Returns transposed m
    return list(map(list, zip(*m)))


def identity(n):
    # Returns square identity matrix of size n
    return [([0] * i) + [1] + ([0] * (n - i - 1)) for i in range(n)]


def zeros(dims):
    # Returns matrix with shape dims filled with 0s
    if type(dims) == int:
        return 0
    if len(dims) == 1:
        return [0.0] * dims[0]
    return [zeros(dims[1:]) for _ in range(dims[0])]


def fill(m, fill_val):
    m_shape = shape(m)
    if len(m_shape) == 1:
        m = [fill_val] * m_shape[0]
        return m
    return [fill(m[m_shape[1:]], fill_val) for _ in range(m_shape[0])]


def random(dims, min_val=0, max_val=1, rng=None):
    # Returns matrix with shape dims filled with
    # random vals between [min_val, max_val)
    if not rng:
        rng = Random()
    if len(dims) == 1:
        return [rng.next(min_val, max_val)] * dims[0]
    return [random(dims[1:], min_val, max_val, rng) for _ in range(dims[0])]


def multiply_accumulate(m, index, a=1, b=0):
    for i in index[:-1]:
        m = m[i]
    m[index[-1]] = a * m[index[-1]] + b
    return m


def multidim_enumerate(m, dim=None):
    if dim is None:
        dim = []
    try:
        for index, m_sub in enumerate(m):
            yield from multidim_enumerate(m_sub, dim + [index])
    except TypeError:
        yield dim, m


def mean(m, axis=None):
    if not axis:
        m_shape = shape(m)
        axis = axis % len(m_shape)
        out = zeros(m_shape[:axis] + m_shape[axis + 1 :])
        for index, val in multidim_enumerate(m):
            out = multiply_accumulate(
                out, index[:axis] + index[axis + 1 :], 1, val / m_shape[axis]
            )
        return out
