from vANNilla.utils.constants import E
from vANNilla.utils.tensor import Tensor


def sigmoid(inputs):
    return Tensor(
        [1 / (1 + (E ** (-Tensor(x)).first_value())) for x in inputs]
    )


def sigmoid_ddx(inputs):
    activated = sigmoid(inputs)
    return [n * (1 - n) for n in activated]


def relu(inputs):
    return [max(0, x) for x in inputs]


def relu_ddx(inputs):
    return [(x > 0) * 1 for x in inputs]


def tanh(inputs):
    return [((E ** x) - (E ** -x)) / ((E ** x) + (E ** -x)) for x in inputs]


def tanh_ddx(inputs):
    return [1 - (x ** 2) for x in inputs]


ACTIVATION_FUNCTIONS = {
    "sigmoid": (sigmoid, sigmoid_ddx),
    "relu": (relu, relu_ddx),
    "tanh": (tanh, tanh_ddx),
}
