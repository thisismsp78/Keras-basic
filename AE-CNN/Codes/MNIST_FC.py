# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:18:30 2018

@author: FaraDars
"""
from keras.datasets import mnist

def plot_history(net_history):
    history = net_history.history
    import matplotlib.pyplot as plt
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['acc']
    val_accuracies = history['val_acc']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['acc', 'val_acc'])

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)

X_train = train_images.reshape(60000, 784)
X_test = test_images.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

from keras.utils import np_utils
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)

#==================================================
# Creating our model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

myModel = Sequential()
myModel.add(Dense(500, activation='relu', input_shape=(784,)))
myModel.add(Dropout(20))
myModel.add(Dense(100, activation='relu'))
myModel.add(Dropout(20))
myModel.add(Dense(10, activation='softmax'))

myModel.summary()
myModel.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

#==================================================
# Train our model
network_history = myModel.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(network_history)

# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)

# Change layers config
myModel.layers[0].name = 'Layer_0'
myModel.layers[0].trainable = False
myModel.layers[0].get_config()