#!/usr/bin/env python

import numpy as np
import tensorflow as tf


input_names = ['x1', 'x2']


def input_fn(num_samples=100, batch_size=100, repeat=True):
    x_data = 2 * np.random.rand(num_samples, 2).astype(np.float32) - 1
    x1 = x_data[:, 0]
    x2 = x_data[:, 1]
    y_data = [xor(xi) for xi in x_data]
    y_data = np.asarray(y_data, dtype=np.float32)
    features = {'x1': x1, 'x2': x2}
    dataset = tf.data.Dataset.from_tensor_slices((features, y_data))
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size)


def xor(x):
    """ Xor function for two inputs """
    if(x[0] >= 0) ^ (x[1] >= 0):
        return 1.0
    return 0.0


# Create feature columns
feature_columns = []
for x in input_names:
    feature_columns.append(tf.feature_column.numeric_column(x))

# Main
estimator = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                      hidden_units=[2],
                                      activation_fn=tf.tanh,
                                      model_dir="/tmp/tensorflow/xor_estimator"
                                      )

estimator.train(input_fn=input_fn, steps=1000)
