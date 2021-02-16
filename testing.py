from vaNNilla import SimpleNetwork
from data import XOR

features, labels = XOR()
network = SimpleNetwork(input_shape=len(features[0]))
network.fit(features, labels, epochs=25000)
predictions = network.predict(features)
[print(feature, round(pred)) for feature, pred in zip(features, predictions)]
