from .utils.activation import ACTIVATION_FUNCTIONS
from .utils.matrix import dot_prod, mean, random, shape, transpose, zeros


class Layer:
    def __init__(self, trainable=True):
        self.input_shape = None
        self.output_shape = None
        self.learning_rate = None
        self.weights = None
        self.biases = None
        self.rng = None
        self.trainable = trainable
        pass

    def initialize(self):
        pass

    def forward(self, inputs):
        pass

    def backward(self, inputs, gradients):
        pass

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        if shape(new_weights) == shape(self.weights):
            self.weights = new_weights
        else:
            raise IndexError(
                "Weight set failed: New weights do not "
                "match shape of old weights"
            )

    def get_biases(self):
        return self.biases

    def set_biases(self, new_biases):
        if shape(new_biases) == shape(self.biases):
            self.weights = new_biases
        raise IndexError(
            "Bias set failed: New biases do not match shape of old biases"
        )


class Activation(Layer):
    def __init__(self, function_name):
        super().__init__(trainable=False)
        if function_name not in ACTIVATION_FUNCTIONS:
            raise RuntimeError(
                f'Activation function "{function_name}" not recognized'
            )
        self.activation, self.activation_ddx = ACTIVATION_FUNCTIONS[
            function_name
        ]

    def forward(self, inputs):
        return self.activation(inputs)

    def backward(self, inputs, gradients):
        return dot_prod(self.activation_ddx(gradients), gradients)


class Dense(Layer):
    def __init__(self, neurons):
        super().__init__()
        self.input_shape = (neurons,)

    def initialize(self):
        xavier_constant = (
            2 / (self.input_shape[0] + self.output_shape[0])
        ) ** 0.5
        self.weights = random(
            dims=(self.input_shape[0], self.output_shape[0]),
            min_val=-xavier_constant,
            max_val=xavier_constant,
        )
        self.biases = zeros(self.output_shape[0])

    def forward(self, inputs):
        return [
            xw + b
            for xw, b in zip(dot_prod(inputs, self.weights), self.biases)
        ]

    def backward(self, inputs, gradient_output):
        gradient_input = dot_prod(gradient_output, transpose(self.weights))
        gradient_weights = dot_prod(transpose(inputs), gradient_output)
        gradient_biases = [
            shape(inputs)[0] * avg for avg in mean(gradient_output, axis=0)
        ]

        self.set_weights(
            [
                weight - self.learning_rate * gradient_weight
                for weight, gradient_weight in zip(
                    self.weights, gradient_weights
                )
            ]
        )
        self.set_biases(
            [
                bias - self.learning_rate * gradient_bias
                for bias, gradient_bias in zip(self.biases, gradient_biases)
            ]
        )

        return gradient_input
