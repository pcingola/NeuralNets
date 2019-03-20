#!/usr/bin/env python

import math
import numpy as np


RAND_WEIGHT_FACTOR = 0.1

# Trainig data
XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_OUTPUTS = np.array([0, 1, 1, 0])

neuron_id = 0


class Neuron:
    '''
    Neuron for backpropagation
    IMPORTANT: This is built for educational pourposes, efficiency is not a concern.
    '''
    def __init__(self, input_neurons=list()):
        ''' Initialzie parameters '''
        global neuron_id
        self.id = neuron_id
        neuron_id += 1
        self.fan_in = len(input_neurons)
        self.out = 0.0
        self.w = RAND_WEIGHT_FACTOR * (2.0 * np.random.rand((self.fan_in)) - 1.0)
        self.input_neurons = input_neurons
        # Error back-propagation
        self.delta = 0.0
        self.sum_delta = 0.0
        self.delta_w = np.zeros((self.fan_in))

    def backprop(self):
        ''' Perform a back-propagation '''
        self.delta = self.sum_delta * self.phi_prime()
        for i, n in enumerate(self.input_neurons):
            n.sum_delta += self.delta * self.w[i]
            self.delta_w[i] = n.out * self.delta

    def calc(self):
        ''' Calculate neuron's output '''
        ins = self.get_inputs()
        h = np.dot(self.w, ins)
        self.out = self.phi(h)

    def connect(self, n):
        ''' Connect neuron's "n" output to this neuron's input '''
        self.input_neurons.append(n)
        return self

    def get_inputs(self):
        ''' Get neuron's inputs as a vector '''
        return np.array([n.out for n in self.input_neurons])

    def phi(self, h):
        ''' Neuron's transfer function '''
        return 1.0 / (1.0 + math.exp(-h))

    def phi_prime(self):
        ''' Neuron's transfer function derivate '''
        return self.out * (1.0 - self.out)

    def update_w(self, eta):
        ''' Update weights '''
        self.w -= eta * self.delta_w

    def __str__(self):
        return f"id: {self.id}, output: {self.out}, weights: {self.w}, delta: {self.delta}, sum_delta: {self.sum_delta}"


def train(layers, data_in, data_out):
    ''' Train a neural network (one iteration) '''
    # For each input sample...
    sum_loss = 0.0
    num_samples = data_in.shape[0]
    neurons = [n for l in layers[1:] for n in l]
    neurons_rev = reversed(list(neurons))
    n_out = layers[2][0]
    eta = 0.1
    for i in range(num_samples):
        # Set network inputs
        for j, n_in in enumerate(layers[0]):
            n_in.out = data_in[i, j]
        # Calculate outputs for all neurons (forward propagation)
        for n in neurons:
            n.calc()
            print(f"\tFW: {n}")
        print(f"FW: {layers[0][0].out} , {layers[0][1].out} => {layers[1][0].out} , {layers[1][1].out} => {n_out.out}")
        # Backpropagation
        for n in neurons:
            n.sum_delta = 0
        diff_loss = (n_out.out - data_out[i])
        n_out.sum_delta = diff_loss
        for n in neurons_rev:
            n.backprop()
            print(f"\tBP: {n}")
            #n.update_w(eta)
        print(f"BP: {layers[0][0].delta} , {layers[0][1].delta} => {layers[1][0].delta} , {layers[1][1].delta} => {n_out.delta}\n")
        # Calculate loss function
        sum_loss += diff_loss * diff_loss
    loss = sum_loss / (2.0 * num_samples)
    print(f"loss = {loss}")
    # Calculate cummulative gradient (backwards propagation)


def create_network():
    ''' Create a fully connected network to solve XOR problem, return layers of neurons '''
    # Create 'placeholders' (inputs and bias)
    bias = Neuron()
    bias.out = 1.0
    in1 = Neuron()
    in2 = Neuron()

    # Create neurons and connect them
    n1 = Neuron([bias, in1, in2])
    n2 = Neuron([bias, in1, in2])
    n_out = Neuron([bias, n1, n2])

    return [[in1, in2], [n1, n2], [n_out]]


# Train
layers = create_network()
for i in range(200):
    train(layers, XOR_INPUTS, XOR_OUTPUTS)
