from __future__ import division
from math import *
import numpy as np
import astroML

#Rectified Linear Unit non-linearity (or ReLU). Used in Neural Networks in place of the sigmoid function. It is simply thresholding at zero:
#Function: f = max (a, 0)

class Unit (object):
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

a = Unit (1.0, 0.0)

step = 1e-2

f = Unit(max (a.value, 0), 0.0)
f.gradient = 1.0
print "original f", f.value

#a > 0 ? 1.0 * f.gradient : 0.0 #syntax of conditional operator in other languages, if a > 0, gradient = 1*df, otherwise gradient = 0

a.gradient = 1.0 * f.gradient if a > 0 else 0.0

a.value = a.value + a.gradient * step

f = Unit(max (a.value, 0), 0.0)

print "f after tug upwards", f.value
print "Yay!! :)"


