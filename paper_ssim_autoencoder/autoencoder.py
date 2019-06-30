#!/usr/bin/env python3

import tensorflow as tf
import glob
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K

from PIL import Image

from sklearn.feature_extraction import image

class SSIM_Autoencoder:
    def __init__(self):
        # Hyperparameters
        self.d = 100
        self.relu_slope = 0.2
        self.c1 = 0.01
        self.c2 = 0.03
        self.kernel_size = 11
        self.x_train = np.array([])
        self.x_test = np.array([])

    def ssim_index(self, y_true, y_pred):

        kernel = [1, self.kernel_size, self.kernel_size, 1]
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
        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)

        denom = ((K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2))
        ssim /= denom

        return ssim


    def load_data(self):
        ix=0
        files_train = glob.glob("./dataset/texture_1/train/good/*.png")
        train_imgs = np.array([np.array(Image.open(fname)) for fname in files_train])
        train_imgs = train_imgs / 255.0
        for train_im in train_imgs:
            patches = image.extract_patches_2d(train_im, (128, 128), 10)
            self.x_train = np.concatenate((self.x_train, patches)) if len(self.x_train) > 0 else patches
        self.x_train = np.reshape(self.x_train, (len(self.x_train), 128, 128, 1))

        files_test = glob.glob("./dataset/texture_1/test/defective/*.png")
        test_imgs = np.array([np.array(Image.open(fname)) for fname in files_test])
        test_imgs = test_imgs / 255.0
        for test_im in test_imgs:
            patches = image.extract_patches_2d(test_im, (128, 128), 10)
            self.x_test = np.concatenate((self.x_test, patches)) if len(self.x_test) > 0 else patches
        self.x_test = np.reshape(self.x_test, (len(self.x_test), 128, 128, 1))

        """
        fig = plt.figure(figsize=(32,32))
        rows=5
        cols=4
        for i in range(1, rows*cols+1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(np.reshape(self.x_test[i], (128,128)))
        plt.show()    
        """

    def train(self):
        ## Encoder

        # Input
        input_img = Input(shape=(128,128,1))

        # Conv1
        x = Conv2D(32, (4,4), strides=(2,2), padding='same')(input_img)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv2
        x = Conv2D(32, (4,4), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv3
        x = Conv2D(32, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv4
        x = Conv2D(64, (4,4), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv5
        x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv6
        x = Conv2D(128, (4,4), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv7
        x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv8
        x = Conv2D(32, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # Conv9
        encoded = Conv2D(self.d, (8,8), strides=(1,1), padding='valid', activation='linear')(x)


        ## Decoder

        # TConv9
        x = Conv2DTranspose(32, (8,8), strides=(1,1), padding='valid')(encoded)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv8
        x = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv7
        x = Conv2DTranspose(128, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv6
        x = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv5
        x = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv4
        x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv3
        x = Conv2DTranspose(32, (3,3), strides=(1,1), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv2
        x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=self.relu_slope)(x)

        # TConv1
        decoded = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='linear')(x)

        autoencoder = Model(input_img, decoded)

        autoencoder.compile(optimizer='adam', loss=self.ssim_index)

        #autoencoder.summary()

        autoencoder.fit(
            self.x_train,
            self.x_train,
            epochs=50,
            batch_size=128,
            shuffle=True,
            validation_data=(self.x_test, self.x_test),
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
        )

        # Save the model for further use
        autoencoder.save("model.h5")
        print("Model successfully saved to disk.")


def main():
    ssim_autoencoder = SSIM_Autoencoder()
    ssim_autoencoder.load_data()
    ssim_autoencoder.train()

if __name__ == '__main__':
    main()  