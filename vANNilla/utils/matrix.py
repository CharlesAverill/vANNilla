def dot_prod(m, n):
    return sum(m_i * n_i for m_i, n_i in zip(m, n))


def outer_prod(m, n):
    if len(m[0]) != len(n):
        raise RuntimeError("Dimension mismatch: M height should equal N width")
    out = []
    for row in m:
        out.append(sum([row[i] * n[i] for i in range(len(n))]))

    return out


def transpose(m):
    return list(map(list, zip(*m)))


def identity(size):
    return [([0] * i) + [1] + ([0] * (size - i - 1)) for i in range(size)]
