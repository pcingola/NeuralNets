#!/usr/bin/env python

# -----------------------------------------------------------------------------
#
# XOR dataset
#
#                                                               Pablo Cingolani
# -----------------------------------------------------------------------------

import math
import numpy as np
import random
import tensorflow as tf

# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 100

# Maximum number of training steps.
# Note: For the XOR 2 input problem we should be able to do it using ~200 epochs
MAX_STEPS = 1000

# Number of samples (input dataset is created randomly)
NUM_SAMPLES = 10

INPUT_NAMES = ['x1', 'x2']


def batch(data_set, batch_size):
    """
    Batch a dataset
    Note: Manual implementation, far from optimal, but easy to understand
    """
    num_samples = data_set[0].shape[0]
    m = (num_samples / batch_size) - 1
    if num_samples % batch_size != 0:
        m += 1
    r = random.randint(0, m)
    rmin, rmax = r * batch_size, (r + 1) * batch_size - 1
    rmax = min(rmax, num_samples - 1)
    return data_set[0][rmin:rmax], data_set[1][rmin:rmax]


def bias_init(num_units):
    """ Initialzie bias tensor """
    return tf.zeros([num_units])


def create_data_set(num_samples=NUM_SAMPLES):
    """
    Create training dataset.
    Note: Numbers are converted to float32 which is the default in tensorflow
    """
    print("Create dataset")
    x_data = (2 * np.random.rand(num_samples, 2) - 1).astype(np.float32)
    y_data = np.asarray([xor(xi) for xi in x_data], np.float32)
    y_data.shape = (num_samples, 1)
    return x_data, y_data


def create_input_features():
    """ Create feature columns """
    feature_columns = []
    for x in INPUT_NAMES:
        feature_columns.append(tf.feature_column.numeric_column(x))
    return feature_columns


def input_fn(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, repeat=True):
    """
        Data input function for Tensorflow's Estimator.
        Creates a Dataset
    """
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


def zeros_init(num_inputs, num_units):
    """ Initialzie weight tensor """
    return tf.zeros([num_inputs, num_units])

def xor(x):
    """ Xor function for two inputs """
    if(x[0] >= 0) ^ (x[1] >= 0):
        return 1.0
    return 0.0


def weight_init(num_inputs, num_units):
    """ Initialzie weight tensor """
    return tf.truncated_normal([num_inputs, num_units], stddev=1.0 / math.sqrt(float(num_inputs)))
