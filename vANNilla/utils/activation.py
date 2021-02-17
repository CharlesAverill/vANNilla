from .constants import E


def sigmoid(x):
    return 1 / (1 + (E ** -x))


def sigmoid_ddx(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return max(0, x)


def relu_ddx(x):
    return (x > 0) * 1


def tanh(x):
    return ((E ** x) - (E ** -x)) / ((E ** x) + (E ** -x))


def tanh_ddx(x):
    return 1 - (x ** 2)


ACTIVATION_FUNCTIONS = {"sigmoid": (sigmoid, sigmoid_ddx),
                        "relu": (relu, relu_ddx),
                        "tanh": (tanh, tanh_ddx)}
