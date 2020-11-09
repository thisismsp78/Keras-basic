# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 20:28:29 2017

@author: FaraDars
"""

'''
Working with tensors in Numpy
'''

import numpy as np

# 0D tensor: Scalar
X0 = np.array(12)
print("X0 dimentions: ", X0.ndim)
print("X0 shape: ", X0.shape)
print("X0 type: ", X0.dtype)

print("--------------------")

# 1D tensor: Vector
X1 = np.array([12.5,3,6.4,4])
print("X1 dimentions: ", X1.ndim)
print("X1 shape: ", X1.shape)
print("X1 type: ", X1.dtype)

print("--------------------")

# 2D tensor: Matrix
X2 = np.array([[1,3,6,4],
              [3,43,1,2],
              [14,5,7,4]])
print("X2 dimentions: ", X2.ndim)
print("X2 shape: ", X2.shape)
print("X2 type: ", X2.dtype)