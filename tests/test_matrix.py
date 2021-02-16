import numpy as np
from vANNilla.utils import outer_prod, dot_prod, transpose


def test_outer_prod():
    m = [[1, 2], [3, 4], [5, 6]]
    n = [5, 6]

    assert np.dot(np.array(m), np.array(n)).tolist() == outer_prod(m, n)


def test_dot_prod():
    m = [1, 2, 3]
    n = [4, 5, 6]

    assert np.dot(np.array(m), np.array(n)) == dot_prod(m, n)


def test_add_to_matrix():
    m = [1, 2, 3]

    assert (np.array(m) + 1.5).tolist() == [m[i] + 1.5 for i in range(len(m))]


def test_transpose():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert np.array(m).T.tolist() == transpose(m)
