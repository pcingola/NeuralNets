#!/usr/bin/env python

#
# Tensorflow example: XOR problem.
#
# Build a neural network to solve XOR problem. Build and train
# manually (create graph and session).
#

import math
import matplotlib.pyplot as plt
import tensorflow as tf

from xor_data import batch, create_data_set, MAX_STEPS, BATCH_SIZE

# Directory to put the training data.
TRAIN_DIR = "/tmp/tensorflow/xor_manual"

# Neural net paramters
IN_SIZE = 2  # Input layer size
HIDDEN1_UNITS = 4  # Number of units in hidden layers.
OUT_SIZE = 1  # Output layer size
beta = 0.3  # Learning rate


def build(nn_graph, use_xentropy=False):
    """ Build the complete graph for feeding inputs, training, and
        saving checkpoints.
    Args:
        nn_graph: An initialized graph
        use_xentropy: If True, loss function will be Mean Corss-entropy.
                      If False, loss functions is Root Mean Squared Error
    Returns:
        init: A hanle to initialize variables
    """
    print("Build graph")
    with nn_graph.as_default():
        # Generate placeholders for the inputs and outputs.
        inputs_placeholder = tf.placeholder(tf.float32)
        outputs_placeholder = tf.placeholder(tf.float32)
        tf.add_to_collection("inputs", inputs_placeholder)
        tf.add_to_collection("outputs", outputs_placeholder)

        # Build a Graph that computes predictions from the inference model.
        nn = nn_create(inputs_placeholder, HIDDEN1_UNITS)
        tf.add_to_collection("nn", nn)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, loss = nn_train(nn, outputs_placeholder, beta, use_xentropy)

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()
    return init, train_op, loss, inputs_placeholder, outputs_placeholder


def nn_create(in_data, hidden1_units):
    """Create a neural network
    Args:
        in_data: Input data placeholder
        hidden1_units: Size of the first hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    # Hidden layer
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IN_SIZE, hidden1_units], stddev=1.0 / math.sqrt(float(IN_SIZE))), name='weights_1')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases_1')
        hidden1 = tf.nn.tanh(tf.matmul(in_data, weights) + biases)

    # Output layer
    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, OUT_SIZE], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights_out')
        biases = tf.Variable(tf.zeros([OUT_SIZE]), name='biases_out')
        output = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)

    # Return graph's last node
    return output


# Build training graph.
def nn_train(nn_out, out, learning_rate, use_xentropy):
    """Build the training graph.
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    if use_xentropy:
        loss_total = -((out * tf.log(nn_out)) + (1 - out) * tf.log(1.0 - nn_out))
    else:
        loss_total = tf.losses.mean_squared_error(nn_out, out)
    loss = tf.reduce_mean(loss_total, name='loss_mean')
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, loss


def train(nn_graph, init, train_op, loss, inputs_placeholder, outputs_placeholder, data_set):
    """ Create a session and train """
    print("Training")
    with tf.Session(graph=nn_graph) as sess:
        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        losses = []
        for step in range(MAX_STEPS):
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
            if step % 100 == 0:
                print('Step %d: loss = %.2f' % (step, loss_value))

        # Write a checkpoint.
        plt.plot(losses)
        plt.show()


# Main
def main_l2():
    data_set = create_data_set()
    nn_graph = tf.Graph()
    init, train_op, loss, inputs_placeholder, outputs_placeholder = build(nn_graph)
    train(nn_graph, init, train_op, loss, inputs_placeholder, outputs_placeholder, data_set)

if __name__ == "__main__":
    main_l2()
