#!/bin/sh

clear
rm -rvf /tmp/tensorflow/linear_fit_custom_model/ 

./src/tensorflow/linear_fit_custom_model.py 
tensorboard --logdir=/tmp/tensorflow/linear_fit_custom_model/

