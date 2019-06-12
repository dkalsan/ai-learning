#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

## Define variables
batch_size = 512
epochs = 20

## Define an auxilliary function
def plot_history(histories, key='sparse_categorical_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()

## Load the digits MNIST dataset
digits_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()

## Save the class names for future reference
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

## Normalize the dataset values
train_images = train_images / 255.0
test_images = test_images / 255.0

# print("Training set shape: {0}".format(train_images.shape))
# print("Training set labels length: {0}".format(len(train_labels)))
# print("Training set labels: {0}".format(train_labels))

# print("Testing set shape: {0}".format(test_images.shape))
# print("Testing set labels length: {0}".format(len(test_labels)))

## Define the layers of the SMALL network
small_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## Compile the model with additional parameters
small_model.compile(optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy', 'sparse_categorical_crossentropy'])

## Inspect the model 
# small_model.summary()

## Train the model
small_history = small_model.fit(
    train_images, 
    train_labels, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images, test_labels)
)

## Define the layers of the MEDIUM network
medium_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## Compile the model with additional parameters
medium_model.compile(optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy', 'sparse_categorical_crossentropy'])

## Train the model
medium_history = medium_model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images, test_labels)
)

## Define the layers of the LARGE network
large_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## Compile the model with additional parameters
large_model.compile(optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy', 'sparse_categorical_crossentropy'])

## Train the model
large_history = large_model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images, test_labels)
)

## Plot the performance
plot_history([('small', small_history), ('medium', medium_history), ('large', large_history)])
