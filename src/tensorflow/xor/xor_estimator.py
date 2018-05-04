#!/usr/bin/env python

import tensorflow as tf
from xor_data import create_input_features, input_fn, MAX_STEPS

#
# Tensorflow example: XOR problem.
#
# Build a neural network to solve XOR problem. Build and train
# using DNNRegressor
#

# Main
estimator = tf.estimator.DNNRegressor(feature_columns=create_input_features(),
                                      hidden_units=[2],
                                      activation_fn=tf.tanh,
                                      model_dir="/tmp/tensorflow/xor_estimator",
                                      optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005)
                                      )

estimator.train(input_fn=input_fn, steps=MAX_STEPS)
