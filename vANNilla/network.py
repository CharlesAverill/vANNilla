from vANNilla.utils.random import Random
from .neuron import Neuron
from .utils.activation import ACTIVATION_FUNCTIONS
from .utils.loss import LOSS_FUNCTIONS
from .utils import outer_prod, transpose, dot_prod


class Network:
    def __init__(self):
        self.rng = Random()
        self.layers = []
        self.biases = []
        self.learning_rate = None


class SimpleNetwork(Network):
    def __init__(self,
                 input_shape,
                 learning_rate=0.05,
                 loss_function='mse'):
        super().__init__()
        self.input_shape = input_shape
        self.layers = [[Neuron(f"Neuron{n}", self.rng.next()) for n in range(input_shape)]]
        self.biases = [0]
        self.learning_rate = learning_rate
        self.activation, self.activation_ddx = ACTIVATION_FUNCTIONS["sigmoid"]
        if loss_function in LOSS_FUNCTIONS:
            self.loss = LOSS_FUNCTIONS[loss_function]
        else:
            raise RuntimeError(f"Activation Function \"{loss_function} is not recognized\"")

    def get_weights(self):
        return [neuron.weight for neuron in self.layers[0]]

    def set_weights(self, new_weights):
        if len(new_weights) != len(self.layers[0]):
            raise RuntimeError("Shape of trained weights does not match shape of input layer")
        for i in range(len(new_weights)):
            self.layers[0][i].weight = new_weights[i]

    def fit(self, features, labels, epochs):
        loss = []
        percent_complete = 0
        for epoch in range(epochs):
            old_weights = self.get_weights()

            predictions = [pred + self.biases[0] for pred in outer_prod(features, old_weights)]
            predictions = self.activation(predictions)

            partial_slope, error = self.loss(predictions, labels, self.activation_ddx)
            loss.append(error)

            transposed = transpose(features)

            new_weights = [weight - self.learning_rate * dot for weight, dot in
                           zip(old_weights, outer_prod(transposed, partial_slope))]
            self.set_weights(new_weights)

            for cut in partial_slope:
                self.biases[0] = self.biases[0] - self.learning_rate * cut

            if epoch >= epochs / (10 * (percent_complete + 1)) and percent_complete / 10 < epoch / epochs:
                percent_complete += 1
                complete = "0" * percent_complete
                incomplete = "-" * (10 - percent_complete)
                print(f"/{complete}{incomplete}/ Loss: {error}")

        return loss

    def predict(self, data):
        if len(data[0]) != self.input_shape:
            raise RuntimeError("Prediction input shape doesn't match network input shape")
        predictions = []
        for row in data:
            predictions.append(self.activation([dot_prod(row, self.get_weights()) + self.biases[0]]))
        return predictions
