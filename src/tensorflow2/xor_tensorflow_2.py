#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], "float32")
y_train = np.array([[-1], [1], [1], [-1]], "float32")

model = Sequential([
    Dense(3, input_dim=2, activation='tanh'),
    Dense(1, activation='tanh')
])

opt = tf.keras.optimizers.Adagrad(lr=0.1)
opt = tf.keras.optimizers.SGD(lr=0.1)
opt = tf.keras.optimizers.Adam(lr=0.1)
opt = tf.keras.optimizers.Adamax(lr=0.1)
opt = tf.keras.optimizers.Nadam(lr=0.1)
model.compile(optimizer=opt, loss='mean_squared_error')

callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]

model.fit(x_train, y_train, epochs=300, verbose=2, callbacks=callbacks)
