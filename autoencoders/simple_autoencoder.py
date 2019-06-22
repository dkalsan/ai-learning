#!/usr/bin/env python3

from __future__ import absolute_import, print_function, division

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt

# encoding dimension
encoding_dim = 32

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
#autoencoder.add(Dense(128, activation='relu', input_shape=(784,)))
autoencoder.add(Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5), input_shape=(784,)))
autoencoder.add(Dense(encoding_dim, activation='linear', name='bottleneck'))
#autoencoder.add(Dense(128, activation='relu'))
autoencoder.add(Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5)))
autoencoder.add(Dense(784, activation='sigmoid'))

## Compile the model with additional parameters
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

## Fit the model
autoencoder.fit(
	train_images,
	train_images,
	epochs=100,
	batch_size=256,
	shuffle=True,
	validation_data=(test_images, test_images)
)

# Extract the encoder
encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)

# Extract/Build the decoder
encoded_input = Input(shape=(encoding_dim,))
decoder = autoencoder.layers[-2](encoded_input)
decoder = autoencoder.layers[-1](decoder)
decoder = Model(encoded_input, decoder)

# Encode and decode some digits
encoded_imgs = encoder.predict(test_images)
decoded_imgs = decoder.predict(encoded_imgs)

# Print mean
print(encoded_imgs.mean())

# Display results
n = 20
plt.figure(figsize=(20,4))
for i in range(n):
	
	# Display original
	ax = plt.subplot(2, n, i+1)
	plt.imshow(test_images[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# Display reconstruction
	ax = plt.subplot(2, n, i+1+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()	

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
#
# We can use ImageDataGenerator to apply random translations, rotations,... as a form of regularization.
# In that case we have to use .fit_generator() instead of .fit() 