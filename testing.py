from vaNNilla import SimpleNetwork
from data import XOR, AND, OR
import matplotlib.pyplot as plt

features, labels = XOR()
network = SimpleNetwork(input_shape=len(features[0]))
history = network.fit(features, labels, epochs=300)
predictions = network.predict(features)
[print(feature, round(pred)) for feature, pred in zip(features, predictions)]

plt.plot(history)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
