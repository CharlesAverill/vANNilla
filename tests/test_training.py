from vaNNilla.utils.activation import ACTIVATION_FUNCTIONS
from vaNNilla.utils.matrix import transpose, outer_prod
import numpy as np


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_ddx_sigmoid(x):
    return np_sigmoid(x) * (1 - np_sigmoid(x))


def test_sigmoid():
    preds = [1, 0, 1, 0, 1]

    assert np_sigmoid(np.array(preds)).tolist() == [ACTIVATION_FUNCTIONS["sigmoid"][0](xw) for xw in preds]


def test_mse():
    activated = [1, 2, 3, 4, 5]
    labels = [1, 0, 6, 0, 10]

    assert (np.array(activated) - np.array(labels)).tolist() == [activated_n - labels_n for activated_n, labels_n in
                                                                 zip(activated, labels)]

    error = (np.array(activated) - np.array(labels)).tolist()
    d_predictions = np_ddx_sigmoid(np.array(activated)).tolist()

    assert (np.array(error) * np.array(d_predictions)).tolist() == [error[i] * d_predictions[i] for i in
                                                                    range(len(error))]


def test_weight_change():
    old_weights = [0, 1]
    learning_rate = 0.05
    inputs = transpose([[1, 2], [2, 3], [3, 4], [5, 6]])
    partial_slope = [.1, -.1, -.5, .5]

    assert (np.array(old_weights) - learning_rate * np.dot(np.array(inputs), np.array(partial_slope))).tolist() == [
        weight - learning_rate * dot for weight, dot
        in
        zip(old_weights,
            outer_prod(inputs, partial_slope))]
