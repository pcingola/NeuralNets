#!/usr/bin/env python

#
# Tensorflow exmaple: XOR using cross-entropy instead of RMS
#

import tensorflow as tf
from xor_manual import build, create_data_set, train


# Main
def main_xentropy():
    data_set = create_data_set()
    nn_graph = tf.Graph()
    init, train_op, loss, inputs_placeholder, outputs_placeholder = build(nn_graph, use_xentropy=True)
    train(nn_graph, init, train_op, loss, inputs_placeholder, outputs_placeholder, data_set)

if __name__ == "__main__":
    main_xentropy()
