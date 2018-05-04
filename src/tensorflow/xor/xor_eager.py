#!/usr/bin/env python

#
# Tensorflow example: XOR problem.
#
# Build a neural network to solve XOR problem. Build and train
# using eager execution
#

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from xor_data import bias_init, create_data_set, weight_init, MAX_STEPS


# Neural net paramters
IN_SIZE = 2  # Input layer size
HIDDEN1_UNITS = 4  # Number of units in hidden layers.
OUT_SIZE = 1  # Output layer size
BETA = 0.3  # Learning rate


class XorEager:
    def __init__(self):
        (x_data, y_data) = create_data_set()
        self.x_data = tf.convert_to_tensor(x_data)
        self.y_data = tf.convert_to_tensor(y_data)
        # Hidden layer
        self.W1 = tfe.Variable(initial_value=weight_init(IN_SIZE, HIDDEN1_UNITS), name="W1")
        self.B1 = tfe.Variable(initial_value=bias_init(HIDDEN1_UNITS), name="B1")
        # Output layer
        self.W2 = tfe.Variable(initial_value=weight_init(HIDDEN1_UNITS, OUT_SIZE), name="W2")
        self.B2 = tfe.Variable(initial_value=bias_init(OUT_SIZE), name="B2")

    def loss(self):
        """ A loss function using mean-squared error """
        error = self.nn() - self.y_data
        return tf.reduce_mean(tf.square(error))

    def grad(self):
        """ Calculate the gradient of loss with respect to weights and biases """
        with tf.GradientTape() as tape:
            tape.watch(self.x_data)
            loss_value = self.loss()
        return tape.gradient(loss_value, [self.W1, self.B1, self.W2, self.B2])

    def nn(self):
        """ Neural network forward propagation """
        l1 = tf.nn.tanh(tf.matmul(self.x_data, self.W1) + self.B1)
        output = tf.nn.sigmoid(tf.matmul(l1, self.W2) + self.B2)
        return output

    def train(self, train_steps=MAX_STEPS, beta=BETA):
        print("Initial loss: {:f}".format(self.loss()))
        for i in range(train_steps):
            dW1, dB1, dW2, dB2 = self.grad()
            self.W1.assign_sub(beta * dW1)
            self.B1.assign_sub(beta * dB1)
            self.W2.assign_sub(beta * dW2)
            self.B2.assign_sub(beta * dB2)
            if i % 100 == 0:
                print("\t{:d}\t{:f}".format(i, self.loss()))
        print("Loss after training: {:f}".format(self.loss()))


# Main
tf.enable_eager_execution()
xor_eager = XorEager()
xor_eager.train()
