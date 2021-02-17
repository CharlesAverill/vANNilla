from .constants import E


def sigmoid(x):
    return 1 / (1 + (E ** -x))


def sigmoid_ddx(x):
    return sigmoid(x) * (1 - sigmoid(x))


ACTIVATION_FUNCTIONS = {"sigmoid": (sigmoid, sigmoid_ddx)}
