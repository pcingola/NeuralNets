#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os

from subprocess import Popen, PIPE, STDOUT
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def find_empty_gpu():
	'''
	Pick an empty GPU: Query using nvidia-smi, parse output and
	return the first available GPU that has no process associated
	'''
	p = Popen(['nvidia-smi', '-q'], stdout=PIPE, stderr=PIPE)
	(stdout, stderr) = p.communicate()
	stdout = stdout.decode("utf-8")
	gpu_number = ''
	for line in stdout.split('\n'):
		# Parse output, format is 'key : value'
		if line.find(':') < 0:
			continue
		(key, value) = [s.strip() for s in line.split(':', maxsplit=1)]
		if key == 'Minor Number':
			gpu_number = value
		if key == 'Processes' and value == 'None' and gpu_number:
			return gpu_number
	return None

def select_empty_gpu():
	''' Find and select the first empty GPU '''
	gpu_number = find_empty_gpu()
	if not gpu_number:
		raise "Could not find an empty GPU"
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
	
def nn(x_train, y_train):
	model = Sequential([
		Dense(3, input_dim=2, activation='tanh'),
		Dense(1, activation='tanh')
	])

	adam = tf.keras.optimizers.Adagrad(lr=0.1)
	model.compile(optimizer=adam, loss='mean_squared_error')

	callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]

	model.fit(x_train, y_train, epochs=300, verbose=2, callbacks=callbacks)



# Main
x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], "float32")
y = np.array([[-1], [1], [1], [-1]], "float32")

select_empty_gpu()
nn(x, y)

