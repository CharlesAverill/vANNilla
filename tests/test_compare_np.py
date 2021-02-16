import numpy as np
from vaNNilla.utils import outer_prod


def test_outer_prod():
    m = [[1, 2], [3, 4], [5, 6]]
    n = [5, 6]

    assert list(np.dot(np.array(m), np.array(n))) == outer_prod(m, n)
