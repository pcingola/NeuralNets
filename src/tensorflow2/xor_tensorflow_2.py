#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from select_gpu import select_empty_gpu


def nn(x_train, y_train):
	model = Sequential([
		Dense(3, input_dim=2, activation='tanh'),
		Dense(1, activation='tanh')
	])
	adam = tf.keras.optimizers.Adagrad(lr=0.1)
	model.compile(optimizer=adam, loss='mean_squared_error')
	callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]
	model.fit(x_train, y_train, epochs=300, verbose=2, callbacks=callbacks)

# Main
x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], "float32")
y = np.array([[-1], [1], [1], [-1]], "float32")

select_empty_gpu()
nn(x, y)

