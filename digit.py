# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:39:36 2012

@author: Magnus Ericmats
"""

from matplotlib.pyplot import imshow, draw, show
from matplotlib.cm import get_cmap
from scipy.io import loadmat
from pylab import sqrt, floor, ceil, zeros, divide, remainder, permutation
from pylab import concatenate, ones
from numpy.core.fromnumeric import reshape
from numpy import dot, exp, log, mat

def displayData(X):
    print "Visualizing"
    m, n = X.shape
    width = round(sqrt(n))
    height = width
    display_rows = int(floor(sqrt(m)))
    display_cols = int(ceil(m/display_rows))

    print "Cell width:", width
    print "Cell height:", height    
    print "Display rows:", display_rows
    print "Display columns:", display_cols
        
    display = zeros((display_rows*height,display_cols*width))

    # Iterate through the training sets, reshape each one and populate
    # the display matrix with the letter matrixes.    
    for xrow in range(0, m):
        rowindex = divide(xrow, display_cols)
        columnindex = remainder(xrow, display_cols)
        rowstart = int(rowindex*height)
        rowend = int((rowindex+1)*height)
        colstart = int(columnindex*width)
        colend = int((columnindex+1)*width)
        display[rowstart:rowend, colstart:colend] = X[xrow,:].reshape(height,width).transpose()
         
    imshow(display, cmap=get_cmap('binary'), interpolation='none')
    
    # Show plot without blocking
    draw()    
    
def loadMatlabData(filename):
    """ 
        Load data from csv file and divide it into parts for
        training, cross validation and test.
    """
   
    # Load the training data
    print "Loading training data..."
    data = loadmat(filename)
    X = data['X']
    y = data['y']
    
    # Randomly select 100 datapoints to display
    sel = permutation(X.shape[0])
    random_columns = sel[0:100]
    
    displayData(X[random_columns,:])
    
    return X,y

def loadWeights(filename):
    print "Loading saved Neural Network parameters..."
    data = loadmat(filename)
    theta1 = data['Theta1']
    theta2 = data['Theta2']
    return theta1, theta2    
    

def sigmoid(z):
    return 1 / (1 + exp(-z))
    
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, Lambda):

    print nn_params[0:hidden_layer_size * input_layer_size].shape
    
    # Reshape the unrolled parameter vector. Remember the bias nodes.
    theta1 = reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                     (hidden_layer_size, input_layer_size + 1))
    theta2 = reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                     (num_labels, hidden_layer_size + 1))    
    
    # Number of training sets                 
    m = X.shape[0]
    
    # Return this
    J = 0
    theta1_grad = zeros(theta1.shape)
    theta2_grad = zeros(theta2.shape)

    # Substitute all ys as the data presumes indexing starts at 1    
    y = y - 1
    
    # From y, craete a matrix with zeros and ones
    Y = zeros((y.shape[0], num_labels))
    for i in range(0, Y.shape[0]):
        Y[i,y[i]] = 1.0

    # Calculate the hypothesis, h (or a3, the activation in layer 3)            
    a1 = concatenate((ones((X.shape[0],1)),X), axis=1)
    z2 = dot(a1, theta1.transpose())
    a2 = concatenate((ones((X.shape[0],1)),sigmoid(z2)), axis=1)    
    z3 = dot(a2, theta2.transpose())
    h = sigmoid(z3)                     # or a3    

    # Calculate J 
    s = (-Y*log(h) - (1.0-Y)*log(1.0-h)).sum()
    J = 1.0/m*s
    
    # Calculate gradients    
    
    # Unroll gradients
    grad = concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis=1)
        
    return J, grad
    
def main():
    input_layer_size = 20 * 20
    hidden_layer_size = 25
    num_labels = 10

    X, y = loadMatlabData('./ex4data1.mat')
    
    theta1, theta2 = loadWeights('./ex4weights.mat')

    # Unroll parameters    
    nn_params = concatenate((theta1.flatten(), theta2.flatten()), axis=1)
    
    Lambda = 0
    
    # Calculate the cost function
    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, Lambda)
    
    print "Cost at parameters (loaded from ex4weights.mat):", J
    print "(this value should be about 0.287629)"
    # Make sure the plots are not closed    
    show()    
    

if __name__ == "__main__":
    main()    
