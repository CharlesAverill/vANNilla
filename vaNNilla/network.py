from .random import Random
from .neuron import Neuron
from .utils.activation import ACTIVATION_FUNCTIONS
from .utils.loss import LOSS_FUNCTIONS
from .utils import outer_prod, transpose, dot_prod


class SimpleNetwork:
    def __init__(self,
                 input_shape,
                 learning_rate=0.05,
                 activation_function='sigmoid',
                 loss_function='mse'):
        self.rng = Random()
        self.input_shape = input_shape
        self.neurons = [Neuron(f"Neuron{n}", self.rng.next()) for n in range(input_shape)]
        self.bias = self.rng.next()
        self.learning_rate = learning_rate
        if activation_function in ACTIVATION_FUNCTIONS:
            self.activation, self.activation_ddx = ACTIVATION_FUNCTIONS[activation_function]
        else:
            raise RuntimeError(f"Activation Function \"{activation_function} is not recognized\"")
        if loss_function in LOSS_FUNCTIONS:
            self.loss = LOSS_FUNCTIONS[loss_function]
        else:
            raise RuntimeError(f"Activation Function \"{activation_function} is not recognized\"")

    def get_weights(self):
        return [neuron.weight for neuron in self.neurons]

    def set_weights(self, new_weights):
        if len(new_weights) != len(self.neurons):
            raise RuntimeError("Shape of trained weights does not match shape of input layer")
        for i in range(len(new_weights)):
            self.neurons[i].weight = new_weights[i]

    def fit(self, features, labels, epochs):
        percent_complete = 0
        for epoch in range(epochs):
            old_weights = self.get_weights()

            dots = outer_prod(features, old_weights)
            predictions = [dot + self.bias for dot in dots]

            partial_slope, error = self.loss(predictions, labels, self.activation, self.activation_ddx)
            transposed_features = transpose(features)

            new_weights = [weight - self.learning_rate * dot for weight, dot
                           in
                           zip(old_weights,
                               outer_prod(transposed_features, partial_slope))]
            self.set_weights(new_weights)

            if epoch >= epochs / (10 * (percent_complete + 1)) and percent_complete / 10 < epoch / epochs:
                percent_complete += 1
                complete = "0" * percent_complete
                incomplete = "-" * (10 - percent_complete)
                print(f"/{complete}{incomplete}/ Loss: {error}")

    def predict(self, data):
        if len(data[0]) != self.input_shape:
            raise RuntimeError("Prediction input shape doesn't match network input shape")
        predictions = []
        for row in data:
            predictions.append(self.activation(dot_prod(row, self.get_weights()) + self.bias))
        return predictions
