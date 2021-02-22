from vANNilla import SimpleNetwork
from data.regression import quadratic_regression, linear_regression
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import kerastroke

features, labels = linear_regression()
'''
tf_model = tf.keras.models.Sequential(
    tf.keras.layers.Dense(len(features[0]), input_shape=(len(features[0]),), activation='sigmoid')
)
tf_model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=.05))
tf_history = tf_model.fit(np.array(features), np.array(labels),
                          epochs=150,
                          callbacks=[kerastroke.Stroke()])
'''
network = SimpleNetwork(input_shape=len(features[0]), learning_rate=.1)
history = network.fit(features, labels, epochs=150)
uniform = [[i / 100.0] for i in list(range(100))]
predictions = network.predict(uniform)
[print(feature, pred, label) for feature, pred, label in zip(features, predictions, labels)]


plt.scatter(features, labels)
plt.plot(uniform, predictions, color='orange', linewidth=3)
plt.legend(["features", "predictions"])
plt.show()


plt.plot(history)
# plt.plot(tf_history.history["loss"])
plt.legend(["SimpleNetwork", "Keras"])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
