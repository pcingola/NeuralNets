#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import dataset_images as ds


class Nn(object):
    '''
    Neural network for image face recognition
    '''

    def __init__(self):
        self.layers_size = [20, 100, 1]
        self.input_size = 400
        self.train_iterations = 10000
        self.show_every = 100
        self.learning_rate = 0.01
        self.tf_log_dir = 'logs'
        # Define neural net variables
        self.x = None
        self.W = []
        self.b = []
        self.h = []
        self.y = []

    def build_model(self):
        # Create input
        self.x = tf.placeholder(tf.float32, [None, self.input_size], name='x')

        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since
        # images are grayscale
        x_image = tf.reshape(self.x, [-1, 20, 20, 1])

        # First convolutional layer - maps one grayscale image to 32
        # feature maps.
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)    # Pooling layer

        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)    # Second pooling layer.

        # Fully connected layer 1 -- after 2 round of downsampling, our image
        # is down to 5x5x64 feature maps -- maps this to 1024 features.
        fc_in_size = 5 * 5 * 64
        W_fc1 = self.weight_variable([fc_in_size, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, fc_in_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents
        # co-adaptation of features.
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = self.weight_variable([1024, 2])
        b_fc2 = self.bias_variable([2])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 2])

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_,
                logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(
            self.cross_entropy)
        self.correct_prediction = tf.equal(
            tf.argmax(self.y_conv, 1),
            tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def load_dataset(self):
        # Import data
        self.dataset = ds.DataSetImages()
        self.dataset.load()
        self.dataset.create_train_test_sets(one_hot=True)
        self.input_size = len(self.dataset.X_train[0])
        print('Input size:', self.input_size)

    def log_open(self, sess):
        self.log_writer = tf.summary.FileWriter(self.tf_log_dir, sess.graph)

    def log_close(self):
        self.log_writer.close()

    def run(self):
        tf.app.run(main=self.run_training, argv=[])

    def run_training(self, args):
        with tf.Graph().as_default():
            self.load_dataset()
            self.build_model()
            with tf.Session() as sess:
                self.train(sess)

    def show_error_files(self, y_d, y):
        files = self.dataset.files_test
        print('Not detected:')
        for i in range(len(y_d)):
            if y_d[i] == 1 and y_d[i] != y[i]:
                print('\t', files[i][0])

    def show_results(self, sess, i):
        '''
        Show training results: Evaluates test dataset
        '''
        (X_test, y_test) = self.dataset.test_set()
        cross_entropy, y_val, accuracy = sess.run(
            [self.cross_entropy, self.y, self.accuracy],
            feed_dict={self.x: X_test, self.y_: y_test, self.keep_prob: 1.0}
            )

        # Show to stderr
        print('\nIteration', i,
              '\ttrain_size:', self.dataset.train_size,
              '\ttest_size:', self.dataset.test_size,
              '\tenergy:', cross_entropy,
              '\taccuracy:', accuracy,
              )

    def show_mark(self):
        sys.stdout.write('.')
        sys.stdout.flush()

    def train(self, sess):
        '''
        Train neural network
        '''
        # Initializa variables
        self.log_open(sess)  # Start tf logger for tensorboard
        tf.global_variables_initializer().run()

        # Train
        for i in range(self.train_iterations):
            self.show_mark()
            batch_xs, batch_ys = self.dataset.next_batch()
            feed_dict = {self.keep_prob: 0.5,
                         self.y_: batch_ys,
                         self.x: batch_xs
                         }
            sess.run([self.train_step], feed_dict=feed_dict)

            if i % self.show_every == 0:
                self.show_results(sess, i)

        self.log_close()


if __name__ == '__main__':
    nn = Nn()
    nn.run()
