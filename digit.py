# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:39:36 2012

@author: Magnus Ericmats
"""

from matplotlib.pyplot import imshow
from matplotlib.cm import get_cmap
from matplotlib.pyplot import draw, show
from scipy.io import loadmat
from pylab import sqrt, floor, ceil, zeros, divide, remainder, permutation
from pylab import concatenate
from numpy.core.fromnumeric import reshape

input_layer_size = 20 * 20
hidden_layer_size = 25
num_labels = 10

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
    

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, Lambda):

    print nn_params[0:hidden_layer_size * input_layer_size].shape
    
    theta1 = reshape(nn_params[0:hidden_layer_size * input_layer_size],
                     hidden_layer_size, input_layer_size)
    theta2 = reshape(nn_params[hidden_layer_size * input_layer_size + 1:],
                               num_labels, hidden_layer_size)    

def main():
    X, y = loadMatlabData('./ex4data1.mat')
    
    theta1, theta2 = loadWeights('./ex4weights.mat')

    # Unroll parameters    
    nn_params = concatenate((theta1.flatten(), theta2.flatten()), axis=1)
    
    Lambda = 0
    
    # Calculate the cost function
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, Lambda)
                       
    # Make sure the plots are not closed    
    show()    
    

if __name__ == "__main__":
    main()    
