from __future__ import division
from math import *
import numpy as np
import astroML


#Learning a support vector machine, very popular type of linear classifier
#Just using binary-labeled 2-D vectors for now, [x1, y1] --> -1 (or 1)
#Use f = ax + by + c, and x and y are fixed inputs, a, b, c are "improved" by tugging (backprop)
x = -2.0
y = 5.0
z = -4.0
step = 1e-2

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
        
class Unit (object):
    def __init__(self, value, gradient):
        self.value = value  #value from forward pass
        self.gradient = gradient #gradient from backward pass

#just automates the single neuron process of tugging
class Circuit (object):
    def __init__(self, a, b, c, x, y):
        self.a = a
        self.b = b
        self.c = c
        self.x = x
        self.y = y
    def forward(self):
        self.multiplication1 = multiplicationGate(a, x)
        self.multiplication2 = multiplicationGate(b, y)
        self.ax = self.multiplication1.forward()
        self.by = self.multiplication2.forward()
        self.addition1 = additionGate(ax, by)
        self.axplusby = self.addition1.forward()
        self.addition2 = additionGate(axplusby, c)
        self.f = addition2.forward()
        return f
        
    def backward(self, gradientTop):
        self.f.gradient = gradientTop
        self.addition2.backward()
        self.addition1.backward()
        self.multiplication2.backward()
        self.multiplication1.backward()
        
class SVM (object):
    def __init__(self, x, y, circuit):
        self.a = Unit(1.0, 0.0) #change these to truly random starting conditions eventually!!
        self.b = Unit(-2.0, 0.0)
        self.c = Unit(-1.0, 0.0)
        self.x = x
        self.y = y
        self.circuit = circuit
        
    def forward(self, x, y): #inputs, but in the form of Units this time
        self.unitOutput = circuit.forward (x, y, self.a, self.b, self.c)
        return self.unitOutput #first guess after 1 tug (?)
    
    #seeing if guess matched label, if not we tug
    def backward(self, label):
        if label == -1 and unitOutput.value > -1: #too high, pull down
            self.tug = -1.0
        if label == 1 and unitOutput.value < 1: #too low, pull up
            self.tug = 1.0
        circuit.backward(self.tug)

    def updateInputs(self):
        step = 0.01
        a = a.value + step * a.gradient
        b = b.value + step * b.gradient
        c = c.value + step * c.gradient

    #run through entire learning iteration
    def learnIteration (x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.updateInputs()
        
#Now train SVM with Stochastic Gradient Descent
#(linear classifier technique, simply adjusting a, b, c given inputs x, y)

N = 6 #number of input vectors
D = 2 #dimension of input vectors

data = np.empty((6, 2))  #7 rows, 1 col, each row is 1 vector (or in this case array)
labels = np.empty ((6,))


##    data[0,0] = 1.2
##    data [0,1] = 0.7
data[0] = np.array([1.2, 0.7])
data[1] = np.array([-0.3, -0.5])
data[2] = np.array([3.0, 0.1])
data[3] = np.array([-0.1, -1.0])
data[4] = np.array([-1.0, 1.1])
data[5] = np.array([2.1, -3])

print data

labels[0] = 1
labels[1] = -1
labels[2] = 1
labels[3] = -1
labels[4] = -1
labels[5] = 1


