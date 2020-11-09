# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:42:31 2017

@author: FaraDars
"""

'''
Deep learning model in Keras
Implementation steps:
    1- Data preparation (Train/Validation/Test)
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Train the model (using fit())
'''

from keras.datasets import mnist

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)

# Data visualization
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap='binary')

# Data manipulation
my_data = train_images[10:100]
my_labels = train_labels[10:100]
my_data = train_images[10:100,:,:]
my_data = train_images[10:100,0:28,0:28]
my_data = train_images[10:100,7:-7,7:-7]
my_data = train_images[:,7:-7,7:-7]

batch1 = train_images[:128]
batch2 = train_images[128:256]
n=10
batch_n = train_images[128*n:128*(n+1)]




