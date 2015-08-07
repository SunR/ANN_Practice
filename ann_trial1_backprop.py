from __future__ import division
from math import *
import numpy as np
import astroML


#Function: f = (x + y)*z
x = -2.0
y = 5.0
z = -4.0
step = 1e-2

def forwardAddGate (a, b):
    return a + b

def forwardMultiplyGate (a, b):
    return a * b

q = forwardAddGate(x, y)
f = forwardMultiplyGate(q, z)
print "first f", f

dq_wrt_x = 1.0 #partial derivative wrt (with respect to) some var
dq_wrt_y = 1.0 #we can then use chain rule on these
df_wrt_q= z
df_wrt_z = q

df_wrt_x = df_wrt_q * dq_wrt_x
df_wrt_y = df_wrt_q * dq_wrt_y

#we now have partial derivatives of f wrt x, y, and z --> we have the "forces"

gradient = np.array([df_wrt_x, df_wrt_y, df_wrt_z])

print "gradient", gradient

#now apply forces to inputs, using something similar to Euler's method

x = x + step * df_wrt_x
y = y + step * df_wrt_y
z = z + step * df_wrt_z

print x, y, z

q = forwardAddGate(x, y)
f = forwardMultiplyGate(q, z)
print "second f", f


