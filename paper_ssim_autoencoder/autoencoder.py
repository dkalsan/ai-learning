#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
d = 100
k = 11
relu_slope = 0.2

## Encoder

# Input
input_img = Input(shape=(128,128,1))

# Conv1
x = Conv2D(32, (4,4), strides=(2,2), padding='same')(input_img)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv2
x = Conv2D(32, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv3
x = Conv2D(32, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv4
x = Conv2D(64, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv5
x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv6
x = Conv2D(128, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv7
x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv8
x = Conv2D(32, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# Conv9
encoded = Conv2D(d, (8,8), strides=(1,1), padding='valid', activation='linear')(x)


## Decoder

# TConv9
x = Conv2DTranspose(32, (8,8), strides=(1,1), padding='valid')(encoded)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv8
x = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv7
x = Conv2DTranspose(128, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv6
x = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv5
x = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv4
x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv3
x = Conv2DTranspose(32, (3,3), strides=(1,1), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv2
x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=relu_slope)(x)

# TConv1
decoded = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='linear')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

"""
autoencoder.fit(
	x_train,
	x_train,
	epochs=50,
	batch_size=128,
	shuffle=True,
	validation_data=(x_test, x_test),
	callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
)
"""