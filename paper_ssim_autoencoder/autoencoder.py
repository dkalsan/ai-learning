#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K

# Hyperparameters
d = 100
k = 11
relu_slope = 0.2
c1 = 0.01
c2 = 0.03
kernel_size = 11

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

def ssim_loss(y_true, y_pred):

	kernel = [1, kernel_size, kernel_size, 1]
	y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_true)[1:]))
	y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

	patches_true = tf.extract_image_patches(y_true, kernel, kernel, [1,1,1,1], 'valid')
	patches_pred = tf.extract_image_patches(y_pred, kernel, kernel, [1,1,1,1], 'valid')

	bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)

	patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
	patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])

	# Get mean
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)

    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)

    # Get std dev
    covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

    # Calculate SSIM
	ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)

	denom = ((K.square(u_true) + K.square(u_pred) + c1) * (var_pred + var_true + c2))
	ssim /= denom

	return ssim

autoencoder.compile(optimizer='adam', loss=ssim_loss)

#autoencoder.summary()

autoencoder.fit(
	x_train,
	x_train,
	epochs=50,
	batch_size=128,
	shuffle=True,
	validation_data=(x_test, x_test),
	callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
)