import numpy as np

from vANNilla.utils import Tensor


def test_comparisons():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([7, 8, 9, 10, 11, 12])
    c = Tensor(100)
    d = Tensor([[100], [101]])

    assert (a == b) is False
    assert (a != b) is True
    assert (c > b > a) is True
    assert (a < b and b < c) is (a < c)
    assert (c <= d) is True
    assert (c >= d) is False


def test_outer_prod():
    m = [[1, 2], [3, 4], [5, 6]]
    n = [5, 6]

    assert (
        Tensor(np.dot(np.array(m), np.array(n)).tolist())
        == Tensor(m) * Tensor(n)
        == Tensor(m).dot(Tensor(n))
    )


def test_add():
    m = [[1, 2, 3], [4, 5, 6]]
    n = [[0, 1, 0], [99, 23, 42.54]]

    assert Tensor((np.array(m) + 1.5).tolist()) == Tensor(m) + 1.5
    assert Tensor((np.array(m) + np.array(n)).tolist()) == Tensor(m) + Tensor(
        n
    )


def test_transpose():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert Tensor(np.array(m).T.tolist()) == Tensor(m).transposed


def test_mean():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    assert Tensor(np.mean(m, axis=0).tolist()) == Tensor(m).mean(axis=0)


def test_first():
    assert Tensor(5).first_value() == 5
    assert Tensor([1, 2, 3]).first_value() == 1
    assert Tensor([[3653, 2345], [235, 6432]]).first_value() == 3653


def test_flat():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    assert Tensor(np.array(m).flatten().tolist()) == Tensor(m).flattened


def test_shape():
    m = [[1, 2, 3], [4, 65]]

    assert np.array(m, dtype=object).shape == Tensor(m).shape


def test_size():
    m = [[1, 2, 3], [4, 65]]

    assert np.array(m, dtype=object).size == Tensor(m).size


def test_tensor_type():
    assert Tensor().tensor_type == "empty"
    assert Tensor(1).tensor_type == "scalar"
    assert Tensor([1, 2, 3]).tensor_type == "vector"
    assert Tensor([1]).tensor_type == "vector"
    assert Tensor([[1, 2, 3], [4, 5, 6]]).tensor_type == "matrix"
    assert Tensor([[1, 2, 3]]).tensor_type == "matrix"
    assert (
        Tensor(list(range(100))).reshape((5, 2, 2, 5)).tensor_type
        == "4-tensor"
    )
