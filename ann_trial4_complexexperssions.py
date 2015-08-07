from __future__ import division
from math import *
import numpy as np
import astroML

#Doing single perceptron gradents in one sweep, combining gates

#Function: f = ax + by + c

class Unit (object):
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

a = Unit (1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)
step = 1e-2

f = Unit(a.value * x.value + b.value * y.value + c.value, 0.0)

print "original f", f.value

f.gradient = 1.0

a.gradient = x.value * f.gradient #df/da = x, standard partial derivs
x.gradient = a.value * f.gradient
b.gradient = y.value * f.gradient
y.gradient = b.value * f.gradient
c.gradient = 1.0 * f.gradient

a.value = a.value + a.gradient * step
b.value = b.value + b.gradient * step
c.value = c.value + c.gradient * step
x.value = x.value + x.gradient * step
y.value = y.value + y.gradient * step

f = Unit(a.value * x.value + b.value * y.value + c.value, 1.0)

print "f after tug upwards", f.value
print "Yay!! :)"


