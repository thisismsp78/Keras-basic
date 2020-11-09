# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:06:11 2018

@author: FaraDars
"""

import glob
import cv2
import numpy as np

# Loading train images
images_path = "C:/Saeed/FaraDars/2/Data/CamVid/train/"
images = glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpg")
images.sort()

X = []
width = 200
height = 100
for img in images:
    image = cv2.imread(img)
    image = cv2.resize(image, (width, height))
    image = image / np.max(image)
    image = image.astype(np.float32)
    X.append(image)
    
    
# loading label images
labels_path = "C:/Saeed/FaraDars/2/Data/CamVid/trainannot/"
labels = glob.glob(labels_path + "*.png") + glob.glob(labels_path + "*.jpg")
labels.sort()

Y = []
out_width = 200
out_height = 100 
nClasses = 12
seg_labels = np.zeros([out_height, out_width, nClasses], dtype='uint8')
for mask in labels:
    label = cv2.imread(mask)
    label = cv2.resize(label, (width, height))
    label = label[:,:,0]
    for c in range(nClasses):
        seg_labels[:,:,c] = (label == c)
    label = label.astype(np.uint8)
    Y.append(label)  