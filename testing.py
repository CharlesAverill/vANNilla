from vANNilla import SimpleNetwork
from data.regression import linear_regression
import matplotlib.pyplot as plt

features, labels = linear_regression()
network = SimpleNetwork(input_shape=len(features[0]), learning_rate=.1)
history = network.fit(features, labels, epochs=600)
uniform = [[i / 100.0] for i in list(range(100))]
predictions = network.predict(uniform)
[print(feature, pred, label) for feature, pred, label in zip(features, predictions, labels)]

plt.scatter(features, labels)
plt.plot(uniform, predictions, color='orange', linewidth=3)
plt.legend(["features", "predictions"])
plt.show()

plt.plot(history)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
