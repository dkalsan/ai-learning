#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

input_img = Input(shape=(28,28,1))

# Conv2D(number_of_output_filters, kernel_size, ...)
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
# Result is a volume of size (28, 28, 16)

# MaxPooling2D(pool_size, strides, ...)
# Stride defaults to pool_size if not specified (2,2)
x = MaxPooling2D((2,2), padding='same')(x)
# Result is a volume of size (14, 14, 16)

x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
# Result is a volume of size (14, 14, 8)

x = MaxPooling2D((2,2), padding='same')(x)
# Result is a volume of size (7, 7, 8)

x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
# Result is a volume of size (7, 7, 8)

encoded = MaxPooling2D((2,2), padding='same')(x)
# Result is a volume of size (4, 4, 8)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(
	x_train,
	x_train,
	epochs=50,
	batch_size=128,
	shuffle=True,
	validation_data=(x_test, x_test),
	callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
)