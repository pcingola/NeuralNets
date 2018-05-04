#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# Declare constants
x = [[40.0]]

# Note that you cannot use 'y = x + 2.' here because x is not a tensor. That's why
# we use tf.add() instead
y = tf.add(x, 1.0)

# Since 'y' was defined as a tensor in the previous step, we can now use an overloaded operator '+'
y = y + 1

print("y: {}".format(y))

# Simple gradient example
w = tfe.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, [w])
print("grad: {}".format(grad))  # => [tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)]
