import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.regression import linear_regression
from vANNilla import SimpleNetwork

features, labels = linear_regression()

tf_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
tf_model.compile(loss="mse", optimizer=tf.optimizers.Adam(learning_rate=0.05))
tf_history = tf_model.fit(np.array(features), np.array(labels), epochs=150)

network = SimpleNetwork(input_shape=len(features[0]), learning_rate=0.05)
history = network.fit(features, labels, epochs=250)
uniform = [[i / 100.0] for i in list(range(100))]
predictions = network.predict(uniform)
tf_preds = tf_model.predict(uniform)
[
    print(feature, pred, label)
    for feature, pred, label in zip(features, predictions, labels)
]

plt.scatter(features, labels)
plt.legend(["features"])
plt.plot(uniform, predictions, color="orange", linewidth=3)
plt.plot(uniform, tf_preds, color="purple", linewidth=3)
plt.legend(["predictions", "tf_preds"])
plt.show()

plt.plot(history)
plt.plot(tf_history.history["loss"])
plt.legend(["SimpleNetwork", "Keras"])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
