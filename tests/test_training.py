import numpy as np

from vANNilla.utils import Tensor
from vANNilla.utils.activation import ACTIVATION_FUNCTIONS


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_ddx_sigmoid(x):
    return np_sigmoid(x) * (1 - np_sigmoid(x))


def test_activation_funcs():
    preds = [-15, 3.14, 1, 6, 1, 0, -1, -1.56, -12, 45.9]
    # sigmoid
    assert [round(n, 5) for n in np_sigmoid(np.array(preds)).tolist()] == [
        round(n, 5) for n in ACTIVATION_FUNCTIONS["sigmoid"][0](preds)
    ]
    # relu
    np_relu = np.array(preds)
    np_relu[np_relu < 0] = 0
    assert np_relu.tolist() == ACTIVATION_FUNCTIONS["relu"][0](preds)
    # tanh
    np_tanh = np.array(preds)
    np_tanh = (np.exp(np_tanh) - np.exp(-np_tanh)) / (
        np.exp(np_tanh) + np.exp(-np_tanh)
    )
    assert [round(n, 5) for n in np_tanh] == [
        round(xw, 5) for xw in ACTIVATION_FUNCTIONS["tanh"][0](preds)
    ]


def test_mse():
    activated = [1, 2, 3, 4, 5]
    labels = [1, 0, 6, 0, 10]

    assert (np.array(activated) - np.array(labels)).tolist() == [
        activated_n - labels_n
        for activated_n, labels_n in zip(activated, labels)
    ]

    error = (np.array(activated) - np.array(labels)).tolist()
    d_predictions = np_ddx_sigmoid(np.array(activated)).tolist()

    assert (np.array(error) * np.array(d_predictions)).tolist() == [
        error[i] * d_predictions[i] for i in range(len(error))
    ]


def test_weight_change():
    old_weights = [0, 1]
    learning_rate = 0.05
    inputs = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]).transposed
    partial_slope = Tensor([0.1, -0.1, -0.5, 0.5])

    print(
        Tensor(
            (
                np.array(old_weights)
                - np.dot(
                    np.array(inputs.tensor_values),
                    np.array(partial_slope.tensor_values),
                )
                * learning_rate
            ).tolist()
        )
    )

    print(
        Tensor(
            [
                Tensor(weight) - dot * learning_rate
                for weight, dot in zip(old_weights, inputs * partial_slope)
            ]
        )
    )

    assert Tensor(
        (
            np.array(old_weights)
            - np.dot(
                np.array(inputs.tensor_values),
                np.array(partial_slope.tensor_values),
            )
            * learning_rate
        ).tolist()
    ) == Tensor(
        [
            Tensor(weight) - dot * learning_rate
            for weight, dot in zip(old_weights, inputs * partial_slope)
        ]
    )
