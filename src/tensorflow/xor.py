#!/usr/bin/env python

#
# Tensorflow exmaple: XOR
#

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from six.moves import xrange
import tensorflow as tf

# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 100

# Input layer size
IN_SIZE = 2

# Number of units in hidden layers.
HIDDEN1_UNITS = 4

# Input layer size
OUT_SIZE = 1

# Maximum number of training steps.
# Note: For the XOR 2 input problem we should be able to do it using ~200 epochs
MAX_STEPS = 500

# Number of samples (input dataset is created randomly)
NUM_SAMPLES = 1000

# Directory to put the training data.
TRAIN_DIR = "/tmp/"

# Learning rate
beta = 0.3


# Get a batch
def batch(data_set, batch_size):
    num_samples = data_set[0].shape[0]
    m = (num_samples / batch_size) - 1
    if num_samples % batch_size != 0:
        m += 1
    r = random.randint(0, m)
    rmin, rmax = r * batch_size, (r + 1) * batch_size - 1
    rmax = min(rmax, num_samples - 1)
    return data_set[0][rmin:rmax], data_set[1][rmin:rmax]


# Create dataset
def create_data_set(num_samples):
    """
    Create training dataset
    """
    x_data = (2 * np.random.rand(num_samples, 2) - 1).astype(np.float32)
    y = [xor(xi) for xi in x_data]
    y_data = np.asarray(y, np.float32)
    y_data.shape = (num_samples, 1)
    print("Input data:\n" + str(np.concatenate((y_data, x_data), axis=1)))
    return x_data, y_data


# Build inference graph.
def nn_inference(in_data, hidden1_units):
    """Build model up to where it may be used for inference.
    Args:
        in_data: Input data placeholder
        hidden1_units: Size of the first hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    # Hidden 1 layer
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IN_SIZE, hidden1_units], stddev=1.0 / math.sqrt(float(IN_SIZE))), name='weights_1')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases_1')
        hidden1 = tf.nn.tanh(tf.matmul(in_data, weights) + biases)

    # Output layer
    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, OUT_SIZE], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights_out')
        biases = tf.Variable(tf.zeros([OUT_SIZE]), name='biases_out')
        output = tf.nn.tanh(tf.matmul(hidden1, weights) + biases)

    # See what we have constructed.
    print("Writing graph to /tmp/inference.pbtxt")
    tf.train.write_graph(tf.get_default_graph().as_graph_def(), "/tmp", "inference.pbtxt", as_text=True)
    return output


# Build training graph.
def nn_training(nn_out, out, learning_rate):
    """Build the training graph.
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    # l2loss = tf.nn.l2_loss((nn_out - out), name='l2loss')
    l2loss = tf.losses.mean_squared_error(nn_out, out)
    loss = tf.reduce_mean(l2loss, name='l2error')
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, loss


def xor(x):
    """ Xor function for two inputs """
    if(x[0] >= 0) ^ (x[1] >= 0):
        return 1.0
    return -1.0


# Get input data
data_set = create_data_set(NUM_SAMPLES)

# Build the complete graph for feeding inputs, training, and saving checkpoints.
print("Build graph")
nn_graph = tf.Graph()
with nn_graph.as_default():
    # Generate placeholders for the inputs and outputs.
    inputs_placeholder = tf.placeholder(tf.float32)
    outputs_placeholder = tf.placeholder(tf.float32)
    tf.add_to_collection("inputs", inputs_placeholder)
    tf.add_to_collection("outputs", outputs_placeholder)

    # Build a Graph that computes predictions from the inference model.
    nn = nn_inference(inputs_placeholder, HIDDEN1_UNITS)
    tf.add_to_collection("nn", nn)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op, loss = nn_training(nn, outputs_placeholder, beta)

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()


# Run training
print("Run training")
with tf.Session(graph=nn_graph) as sess:
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    losses = []
    for step in xrange(MAX_STEPS):
        # Read a batch.
        batch_in, batch_out = batch(data_set, BATCH_SIZE)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict={inputs_placeholder: batch_in, outputs_placeholder: batch_out})

        losses.append(loss_value)

        # Print out loss value.
        if step % 1000 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))

    # Write a checkpoint.
    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)
    plt.plot(losses)
    plt.show()
