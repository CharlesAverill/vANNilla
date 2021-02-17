from vANNilla import SimpleNetwork
from data import XOR, AND, OR, linear_regression
import matplotlib.pyplot as plt

features, labels = linear_regression()
network = SimpleNetwork(input_shape=len(features[0]), learning_rate=.1)
history = network.fit(features, labels, epochs=600)
predictions = network.predict(features)
[print(feature, pred, label) for feature, pred, label in zip(features, predictions, labels)]

plt.scatter(features, labels)
plt.scatter(predictions, labels)
plt.legend(["features", "predictions"])
plt.show()

plt.plot(history)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
