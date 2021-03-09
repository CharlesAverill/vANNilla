import timeit

preparation = "from vANNilla import SimpleNetwork\n" \
              "from vANNilla.utils import Tensor\n" \
              "from data.regression import linear_regression\n" \
              "features, labels = linear_regression()\n" \
              "network = SimpleNetwork(input_shape=len(features[0]), learning_rate=0.05)\n" \
              "features, labels = Tensor(features), Tensor(labels)\n"

feed_forward = preparation + "fed_forward = network.feed_forward(features)\n"
backpropagate = feed_forward + "loss = network.backpropagate(fed_forward, features, labels)\n"

time_fed_forward = timeit.timeit(feed_forward, number=10)
time_backprop = timeit.timeit(backpropagate, number=10) - time_fed_forward
print("Feed forward:", time_fed_forward)
print("Backpropagate:", time_backprop)
