from .utils.activation import ACTIVATION_FUNCTIONS
from .utils.matrix import outer_prod


class Layer:
    def __init__(self):
        pass

    def forward(self, inputs):
        pass

    def backward(self, inputs, gradients):
        pass


class Activation(Layer):
    def __init__(self, function_name):
        super().__init__()
        if function_name not in ACTIVATION_FUNCTIONS:
            raise RuntimeError(f"Activation function \"{function_name}\" not recognized")
        self.activation, self.activation_ddx = ACTIVATION_FUNCTIONS[function_name]

    def forward(self, inputs):
        return self.activation(inputs)

    def backward(self, inputs, gradients):
        return outer_prod(self.activation_ddx(gradients), gradients)
