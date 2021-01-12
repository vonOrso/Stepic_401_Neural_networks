import numpy as np
import math
import pandas as pd
# import tensorflow as tf
# import tensorflow.keras as keras

# 3.1.6
# x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0 , 1, 1, 0])
# model = keras.Sequential([
#     keras.layers.Dense(2, activation=tf.math.sigmoid),
#     keras.layers.Dense(1, activation=tf.math.sigmoid)
# ])
# model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x, y, epochs=1000)
# print(model.get_weights())

# 3.2.5
# def get_error(deltas, sums, weights):
#     """
#     compute error on the previous layer of network
#     deltas - ndarray of shape (n, n_{l+1})
#     sums - ndarray of shape (n, n_l)
#     weights - ndarray of shape (n_{l+1}, n_l)
#     """
#     d = deltas.dot(weights)*sigmoid_prime(sums)
#     return d.mean(axis=0)