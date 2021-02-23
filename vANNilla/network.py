from vANNilla.layers import Dense
from vANNilla.utils import dot_prod, scalar_dot, transpose
from vANNilla.utils.activation import ACTIVATION_FUNCTIONS
from vANNilla.utils.loss import LOSS_FUNCTIONS
from vANNilla.utils.random import Random


class Network:
    def __init__(self):
        self.rng = Random()
        self.layers = []
        self.biases = []
        self.learning_rate = None

    def get_weights(self):
        pass

    def set_weights(self, new_weights):
        pass

    def fit(self, features, labels, epochs):
        pass

    def predict(self, data):
        pass

    def feed_forward(self, features):
        pass

    def backpropagate(self, data_fed_forward, features, labels):
        pass


class SimpleNetwork(Network):
    """
    A single-layer NN (Dense + Activation)
    """

    def __init__(self, input_shape, learning_rate=0.05, loss_function="mse"):
        super().__init__()
        self.input_shape = input_shape
        self.layer = Dense(neurons=input_shape)
        self.layer.output_shape = (1,)
        self.layer.initialize()
        self.bias = self.layer.biases
        self.learning_rate = learning_rate
        self.activation, self.activation_ddx = ACTIVATION_FUNCTIONS["sigmoid"]
        if loss_function in LOSS_FUNCTIONS:
            self.loss = LOSS_FUNCTIONS[loss_function]
        else:
            raise RuntimeError(
                f'Activation Function "{loss_function} is not recognized"'
            )

    def fit(self, features, labels, epochs):
        loss = []
        percent_complete = 0
        for epoch in range(epochs):
            predictions = self.feed_forward(features)

            loss.append(self.backpropagate(predictions, features, labels))

            if (
                epoch >= epochs / (10 * (percent_complete + 1))
                and percent_complete / 10 < epoch / epochs
            ):
                percent_complete += 1
                complete = "0" * percent_complete
                incomplete = "-" * (10 - percent_complete)
                print(f"/{complete}{incomplete}/ Loss: {loss[-1]}")

        return loss

    def feed_forward(self, features):
        predictions = [
            pred + self.bias
            for pred in dot_prod(features, self.layer.get_weights()[0])
        ]
        return self.activation(predictions)

    def backpropagate(self, data_fed_forward, features, labels):
        partial_slope, error = self.loss(
            data_fed_forward, labels, self.activation_ddx
        )

        transposed = transpose(features)

        old_weights = self.layer.get_weights()[0]
        new_weights = [
            weight - self.learning_rate * dot
            for weight, dot in zip(
                old_weights, dot_prod(transposed, partial_slope)
            )
        ]
        self.layer.set_weights([new_weights])

        for cut in partial_slope:
            self.bias -= self.learning_rate * cut

        return error

    def predict(self, data):
        if len(data[0]) != self.input_shape:
            raise RuntimeError(
                "Prediction input shape doesn't match network input shape"
            )
        predictions = []
        for row in data:
            predictions.append(
                self.activation(
                    [scalar_dot(row, self.layer.get_weights()[0]) + self.bias]
                )
            )
        return predictions
