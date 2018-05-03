#!/usr/bin/env python

#
# Simple Tensorflow example to solve "linear fit" problem
# implementing an 'Estimator'
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SAVE_EVERY_N_STEPS = 1


def input_fn(num_samples=1000, show=False, batch_size=10, repeat=True):
    """ Create input data using NumPy. y = x * 0.1 + 0.3 + noise """
    x_data = np.random.rand(num_samples).astype(np.float32)
    noise = np.random.normal(scale=0.01, size=len(x_data)).astype(np.float32)
    y_data = x_data * 0.1 + 0.3 + noise

    if show:
        plt.plot(x_data, y_data, '.')
        plt.show()

    # Build dataset, repeat and batch
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size)


def model_fn(features, labels, mode, params):
    """Trivial linear model"""
    # Create Variables W and b that compute y_data = W * x_data + b
    with tf.name_scope("linear_model"):
        x_data = features
        y_data = labels
        W = tf.Variable(tf.random_uniform([1], 0.0, 1.0), name='W')
        b = tf.Variable(tf.zeros([1]), name='b')
        y = W * x_data + b

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        print("model_fn: Predict")
        predictions = {'y_hat': y}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluate model
    with tf.name_scope("eval"):
        loss = tf.reduce_mean(tf.square(y - y_data))
        rms = tf.metrics.root_mean_squared_error(labels=labels, predictions=y, name='rms')
        tf.summary.scalar('rms', rms[1])
        metrics = {'rms': rms, 'loss': loss}
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Train model
    assert mode == tf.estimator.ModeKeys.TRAIN
    print("model_fn: Train")
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(0.1)  # Create an optimizer.
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  # Create an operation that minimizes loss.
    return tf.estimator.EstimatorSpec(mode, predictions=y, loss=loss, train_op=train_op)


# Main

# Build estimator
model_dir = "/tmp/tensorflow/linear_fit_custom_model"
classifier = tf.estimator.Estimator(model_fn=model_fn, params={}, model_dir=model_dir)

# Train the model
# Note: One train iteration is OK, we do more just to be able to see a graph in tensorboard
for i in range(10):
    classifier.train(input_fn=lambda: input_fn(num_samples=1000), steps=10)
