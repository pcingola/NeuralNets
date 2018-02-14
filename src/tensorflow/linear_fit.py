#!/usr/bin/env python

#
# Simple Tensorflow example to solve "linear fit" problem
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1.2 Create input data using NumPy. y = x * 0.1 + 0.3 + noise
x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_data))
y_data = x_data * 0.1 + 0.3 + noise

plt.plot(x_data, y_data, '.')
plt.show()

# 1.3 Buld inference graph.
# Create Variables W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], 0.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
y = W * x_data + b

print('W:', W)
print('b:', b)

# 1.4 Build training graph.
loss = tf.reduce_mean(tf.square(y - y_data))  # Create an operation that calculates loss.
optimizer = tf.train.GradientDescentOptimizer(0.5)  # Create an optimizer.
train = optimizer.minimize(loss)  # Create an operation that minimizes loss.
init = tf.initialize_all_variables()  # Create an operation initializes all the variables.

# Uncomment the following 3 lines to see what 'loss', 'optimizer' and 'train' are.
print("loss:", loss)
print("optimizer:", optimizer)
print("train:", train)
print(init)

print(tf.get_default_graph().as_graph_def())

# 1.6 Create a session and launch the graph.
sess = tf.Session()
sess.run(init)
y_initial_values = sess.run(y)  # Save initial values for plotting later.

# See the initial W and b values.
print(sess.run([W, b]))

# 1.7 Perform training.
for step in range(201):
    sess.run(train)
    # See training happen real time.
    if step % 20 == 0:
        print(step, sess.run([W, b]))

print(sess.run([W, b]))

# 1.8 Compare.
plt.plot(x_data, y_data, '.', label="target_values")
plt.plot(x_data, y_initial_values, ".", label="initial_values")
plt.plot(x_data, sess.run(y), ".", label="trained_values")
plt.legend()
plt.ylim(0, 1.0)
plt.show()
