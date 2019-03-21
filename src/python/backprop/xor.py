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
        ins = np.array([n.out for n in self.input_neurons])
        h = np.dot(self.w, ins)
        self.out = self.phi(h)

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


class Network:
    ''' A neural network to solve the XOR problem '''
    def __init__(self, data_in, data_out):
        ''' Create a fully connected network to solve XOR problem, return layers of neurons '''
        self.data_in = data_in
        self.data_out = data_out
        # Create 'placeholders' (inputs and bias)
        bias = Neuron()
        bias.out = 1.0
        in1 = Neuron()
        in2 = Neuron()
        # Create neurons and connect them
        n1 = Neuron([bias, in1, in2])
        n2 = Neuron([bias, in1, in2])
        self.n_out = Neuron([bias, n1, n2])
        # Create layers
        self.layers = [[in1, in2], [n1, n2], [self.n_out]]
        self.neurons = [n for l in self.layers[1:] for n in l]
        self.neurons_rev = self.neurons.copy()
        reversed(self.neurons_rev)

    def backprop(self, num_in):
        # Reset delta_sum
        for n in self.neurons:
            n.sum_delta = 0
        # Calculate output delta
        diff_loss = (self.n_out.out - self.data_out[num_in])
        self.n_out.sum_delta = diff_loss
        # Backpropagation
        for n in self.neurons_rev:
            n.backprop()
            print(f"\tBP: {n}")
            #n.update_w(eta)
        print(f"BP: {self.layers[0][0].delta} , {self.layers[0][1].delta} => {self.layers[1][0].delta} , {self.layers[1][1].delta} => {self.n_out.delta}\n")
        return diff_loss * diff_loss

    def calc(self):
        # Calculate outputs for all neurons (forward propagation)
        for n in self.neurons:
            n.calc()
            print(f"\tFW: {n}")
        print(f"FW: {self.layers[0][0].out} , {self.layers[0][1].out} => {self.layers[1][0].out} , {self.layers[1][1].out} => {self.n_out.out}")

    def set_inputs(self, num_in):
        ''' Set netwok inputs with sample number 'num_in' '''
        for j, n_in in enumerate(self.layers[0]):
            n_in.out = self.data_in[num_in, j]

    def train(self):
        ''' Train a neural network (one iteration) '''
        # For each input sample...
        sum_loss = 0.0
        num_samples = self.data_in.shape[0]
        n_out = self.layers[2][0]
        eta = 0.1
        for i in range(num_samples):
            self.set_inputs(i)
            self.calc()
            sum_loss += self.backprop(i)
        loss = sum_loss / (2.0 * num_samples)
        print(f"loss = {loss}")
        # Calculate cummulative gradient (backwards propagation)


# Train
net = Network(XOR_INPUTS, XOR_OUTPUTS)
net.train()
