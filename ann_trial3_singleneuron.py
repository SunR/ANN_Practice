from __future__ import division
from math import *
import numpy as np
import astroML

#IF BIGGER ANN OR SVM CODE DOESN'T WORK, CHECK THESE ANALYTICAL GRADIENTS WITH NUMERICAL GRADIENTS

#Function: f = sigma(ax + by+c)

#class for each value going through each gate
class Unit (object):
    def __init__(self, value, gradient):
        self.value = value  #value from forward pass
        self.gradient = gradient #gradient from backward pass

class multiplicationGate (object):

    def __init__(self, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2 #check to see if this is actually necessary in python (comment out)
        
    def forward (self): #check if all arguments are necessary
        self.unitAbove = Unit(self.unit1.value * self.unit2.value, 0.0) #is the "self" necessary?
        #unitAbove = Unit(1.0, 1.0)
        return self.unitAbove
        
    def backward (self):
        #computing derivatives for this specific case, where dx=y*dt and dy=x*dt
        self.unit1.gradient = self.unit2.value * self.unitAbove.gradient 
        self.unit2.gradient = self.unit1.value * self.unitAbove.gradient

class additionGate (object):
    def __init__ (self, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2
        
    def forward(self):
        self.unitAbove = Unit (self.unit1.value + self. unit2.value, 0.0) #gradient values will be calculated in the backprop part
        return self.unitAbove
    
    def backward(self):
        self.unit1.gradient = 1.0 * self.unitAbove.gradient #derivative of addition function is just 1
        self.unit2.gradient = 1.0 * self.unitAbove.gradient 

class sigmoidGate (object):
    def __init__(self, unit1):
        self.unit1 = unit1
        
    def forward (self):
        self.unitAbove = Unit ((1.0/(1.0 + exp(-self.unit1.value))), 0.0)
        print self.unitAbove
        return self.unitAbove
    
    def backward (self):
        print "backward sigmoid gate"
        print self.unitAbove.gradient
        self.unit1.gradient = self.unitAbove.value * (1 - self.unitAbove.value) * self.unitAbove.gradient #calculated derivative of sigmoid funtion

#five inputs
a = Unit(1.0, 0.0)
print "a value:", a.value
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)
step = 1e-2

f = (1.0/(1.0 + exp(-x.value)) * (a.value  * x.value + b.value * y.value + c.value))
print "original f", f

#first calculate forwards
multiplication1 = multiplicationGate(a, x)
multiplication2 = multiplicationGate(b, y)
ax = multiplication1.forward()
print type(ax)
print type (a)
print "ax value:", ax.value
by = multiplication2.forward()
addition1 = additionGate(ax, by)
axplusby = addition1.forward()
addition2 = additionGate(axplusby, c)
fNoSigma = addition2.forward()
sigmoid1 = sigmoidGate(fNoSigma)
print "point 1"
f = sigmoid1.forward()

print "point 2"
print f.value

#then caluclate gradients, working backwards from order used to calculate values 

f.gradient = 1.0 #tugging output upwards slightly; to pull down, use -1. Making this value hus (500) results in huge tugs. Is this normal? 
sigmoid1.backward()
print fNoSigma.gradient
addition2.backward() #writes gradients back into all five input Units
addition1.backward()
multiplication2.backward()
multiplication1.backward()

print c.gradient

a.value += step * a.gradient
b.value += step * b.gradient
c.value += step * c.gradient
x.value += step * x.gradient
y.value += step * y.gradient

print a.value, b.value, c.value, x.value, y.value

f = (1.0/(1.0 + exp(-x.value)) * (a.value  * x.value + b.value * y.value + c.value))

print "f after tugging:", f
    


#then apply gradients to inputs


