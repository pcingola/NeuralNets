#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import dataset_images as ds


class Nn(object):
    '''
    Neural network for image face recognition
    '''

    def __init__(self):
        self.layers_size = [1]
        self.input_size = 400
        self.train_iterations = 1000
        self.show_every = 100
        self.learning_rate = 0.005
        self.tf_log_dir = 'logs'
        # Define neural net variables
        self.x = None
        self.W = []
        self.b = []
        self.h = []
        self.y = []

    def build_model(self):
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, self.input_size], name='x')

        layer_num = 0
        lsize_prev = self.input_size
        for lsize in self.layers_size:
            self.create_layer(layer_num, lsize, lsize_prev)
            lsize_prev = lsize
            layer_num = layer_num + 1

        # Define loss and optimizer
        ll = self.last_layer_num()
        self.y_ = tf.placeholder(tf.float32,
                                 [None, self.layers_size[ll]],
                                 name='y_')

        # Energy minimization
        self.energy = tf.nn.l2_loss(self.y[ll] - self.y_,
                                    name="energy")

        # Train
        self.train_step = tf.train. \
            GradientDescentOptimizer(self.learning_rate). \
            minimize(self.energy)

        # Log summaries
        self.summary_energy = tf.summary.scalar('summ_energy', self.energy)
        self.summaries = tf.summary.merge_all()

    def create_bias(self, size, name):
        # return tf.Variable(tf.zeros([size]), name=name)
        return tf.Variable(tf.random_uniform([size], minval=-0.1, maxval=0.1),
                           name=name)

    def create_layer(self, layer_num, lsize, lsize_prev):
        '''
        Create a layer
        '''
        print('Layer ', layer_num, ', size ', lsize)
        with tf.name_scope('layer_' + str(layer_num)):
            # Names
            wname = 'W_' + str(layer_num)
            bname = 'b_' + str(layer_num)
            hname = 'h_' + str(layer_num)
            yname = 'y_' + str(layer_num)

            # Create weight
            self.W.append(self.create_weight(lsize_prev, lsize, wname))

            # Create bias
            self.b.append(self.create_bias(lsize, bname))

            # Create activation
            x = self.x  # Layer input is network input
            if layer_num > 0:   # ... or input from previous layer
                x = self.y[layer_num - 1]
            self.h.append(tf.add(tf.matmul(x, self.W[layer_num]),
                          self.b[layer_num],
                          name=hname)
                          )

            # Create output
            self.y.append(tf.nn.sigmoid(self.h[layer_num], name=yname))

    def create_weight(self, sizex, sizey, name):
        # return tf.Variable(tf.zeros([sizex, sizey]), name=name)
        return tf.Variable(tf.random_uniform([sizex, sizey],
                           minval=-0.1,
                           maxval=0.1),
                           name=name
                           )

    def last_layer_num(self):
        return len(self.layers_size) - 1

    def load_dataset(self):
        # Import data
        self.dataset = ds.DataSetImages()
        self.dataset.load()
        self.dataset.create_train_test_sets()
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

    def show_results(self, sess, i, show_files=False):
        '''
        Show training results: Evaluates test dataset
        '''
        lens = 25
        (X_test, y_test) = self.dataset.test_set()
        test_size = len(X_test)
        y = self.y[self.last_layer_num()]
        energy_val, y_val, summary = sess.run(
            [self.energy, y, self.summaries],
            feed_dict={self.x: X_test, self.y_: y_test}
            )
        y_pred = np.array(y_val.T[0] + 0.5, dtype=int)
        y_d = np.array(y_test.T[0] + 0.5, dtype=int)
        perc = 1.0 - np.linalg.norm(y_pred - y_d, ord=1) / test_size

        # Write data to tensorboard's log
        self.log_writer.add_summary(summary, i)
        self.log_writer.flush()

        # Show to stderr
        print('Iteration', i,
              '\ttrain_size:', self.dataset.train_size,
              '\ttest_size:', self.dataset.test_size,
              '\tenergy:', energy_val,
              '\tperc:', perc,
              '\n\ty :', y_pred[0:lens],
              '\n\ty_:', y_d[0:lens],
              )
        if show_files:
            self.show_error_files(y_d, y_pred)

    def train(self, sess):
        '''
        Train neural network
        '''
        # Initializa variables
        self.log_open(sess)  # Start tf logger for tensorboard
        tf.global_variables_initializer().run()

        # Train
        for i in range(self.train_iterations):
            batch_xs, batch_ys = self.dataset.next_batch()
            _, summary = sess.run(
                [self.train_step, self.summaries],
                feed_dict={self.x: batch_xs, self.y_: batch_ys})

            self.log_writer.add_summary(summary, i)

            if i % self.show_every == 0:
                self.show_results(sess, i)

        self.show_results(sess, i, show_files=True)
        self.log_close()


if __name__ == '__main__':
    nn = Nn()
    nn.run()
