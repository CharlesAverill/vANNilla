import numpy as np

from vANNilla.utils import scalar_dot


def test_scalar_dot():
    m = [1, 2, 3]
    n = [4, 5, 6]

    assert np.dot(np.array(m), np.array(n)) == scalar_dot(m, n)
