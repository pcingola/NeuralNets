#!/usr/bin/env python

#
# Tensorflow example: XOR problem.
#
# Build a neural network to solve XOR problem. Build and train
# using eager execution
#

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from xor_data import create_data_set, weight_init, zeros_init, MAX_STEPS


# Neural net paramters
IN_SIZE = 2  # Input layer size
HIDDEN1_UNITS = 4  # Number of units in hidden layers.
OUT_SIZE = 1  # Output layer size
BETA = 0.3  # Learning rate
ALPHA = 0.9  # Momentum
LAYERS = [IN_SIZE, HIDDEN1_UNITS, OUT_SIZE]


class XorEager:
    def __init__(self, layer_size=LAYERS, beta=BETA, alpha=ALPHA):
        self.alpha = alpha
        self.beta = beta
        (x_data, y_data) = create_data_set()
        self.x_data = tf.convert_to_tensor(x_data)
        self.y_data = tf.convert_to_tensor(y_data)

        self.layer_size = layer_size
        self.num_layers = len(layer_size)
        self.W = []
        self.B = []
        self.speedW = []
        self.speedB = []
        self.learn_variables = []
        self.speed = []
        # Note that layer[0] = input layer
        for i in range(1, self.num_layers):
            self._add_layer(layer_size[i], layer_size[i - 1], i)

    def _add_layer(self, num_units, num_units_prev, layer_num):
        """ Initialize and add a layer """
        sl = str(layer_num)
        w = tfe.Variable(initial_value=weight_init(num_units, num_units_prev), name="W" + sl)
        b = tfe.Variable(initial_value=zeros_init(num_units), name="B" + sl)
        speedW = tfe.Variable(zeros_init(num_units, num_units_prev), name="speed_W" + sl)
        speedB = tfe.Variable(zeros_init(num_units), name="speed_B" + sl)
        self.W.append(w)
        self.B.append(b)
        self.learn_variables.append(w)
        self.learn_variables.append(b)
        self.speed.append(speedW)
        self.speed.append(speedB)

    def loss(self):
        """ A loss function using mean-squared error """
        error = self.nn() - self.y_data
        return tf.reduce_mean(tf.square(error))

    def grad(self):
        """ Calculate the gradient of loss with respect to weights and biases """
        with tf.GradientTape() as tape:
            tape.watch(self.x_data)
            loss_value = self.loss()
        return tape.gradient(loss_value, self.learn_variables)

    def learn(self):
        """ Calculate the learning vector """
        gv = self.grad()
        for i in range(0, len(self.learn_variables)):
            # Calculate speed:
            #     v = alpha * v - beta * gradient
            #     => v -= (1-alpha) * v + beta * gradient
            self.speed[i].assign_sub(self.alpha * self.speed[i] + self.beta * gv[i])
            # Learn: w += v
            self.learn_variables[i].assign_add(self.speed[i])

    def nn(self):
        """ Neural network forward propagation """
        l = self.x_data
        for i in range(0, len(self.W)):
            l = tf.nn.tanh(tf.matmul(l, self.W[i]) + self.B[i])
        return l

    def train(self, train_steps=MAX_STEPS):
        """ Train the neural network using 'simple gradient descent' """
        print("Start loss: {:f}".format(self.loss()))
        for i in range(train_steps):
            self.learn()
            if i % 100 == 0:
                print("\t{:d}\t{:f}".format(i, self.loss()))
        print("End loss: {:f}".format(self.loss()))


# Main
tf.enable_eager_execution()
xor_eager = XorEager()
xor_eager.train()
