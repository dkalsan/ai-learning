#!/usr/bin/env python3

from __future__ import absolute_import, print_function, division

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt

## Load the MNIST dataset
digits_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()

## Normalize dataset values
train_images = train_images / 255.0
test_images = test_images / 255.0

## Preprocess data (reshape 28x28 images to vectors of size 784)
train_images = train_images.reshape((len(train_images), np.prod(train_images.shape[1:])))
test_images = test_images.reshape((len(test_images), np.prod(test_images.shape[1:])))

## Build the model
autoencoder = Sequential()
autoencoder.add(Dense(128, activation='relu', input_shape=(784,)))
autoencoder.add(Dense(32, activation='linear', name='bottleneck'))
autoencoder.add(Dense(128, activation='relu'))
autoencoder.add(Dense(784, activation='sigmoid'))

## Compile the model with additional parameters
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

## Fit the model
autoencoder.fit(
	train_images,
	train_images,
	epochs=50,
	batch_size=256,
	shuffle=True,
	validation_data=(test_images, test_images)
)



## Notes:
# If you want to just preprocess the image, use numpy flatten (avoids unneccessary overhead in the model)
# If you already have a tensor layer or need it inside the model, use keras flatten layer
#
# Tensor is actually a function f(x) that hasn't been evaluated yet.
# We can pass it the x, e.g.: 
# Dense(32, activation='relu')(Input(shape(784,)))
# f ~ Dense(32, activation='relu'), x ~ Input(shape(784,))
# This is called creating a Tensor using a "functional API"
#
# Keras Model:
# model = Model(inputs=[...], outputs=[...])