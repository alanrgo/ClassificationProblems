# This assignment was conducted in a partnership with Mai Jack
# on CSC321 course at UofT

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import random


import cPickle

import os
from scipy.io import loadmat

# Change to your local working directory to make it work.
dir = "/home/jack/Desktop/321A2/"
#dir = "D:/OneDrive/Study/2016w/csc321/a2/"
graph_count = 1

#Load the MNIST digit data
M = loadmat(dir+"mnist_all.mat")

# Rules: in this file usually,
# W's and Wb's passed between functions are of shape n x m where n>=m
# b's, x's, h's, and o's are of shape 10 x 1
# actual y's are exact answers, an int
# Variables above could be stored and passed in lists.


def savefigure():
    '''
    Save the image. Increase global graph_count to avoid overwriting.
    '''
    global graph_count
    savefig(dir+'figure'+str(graph_count)+'.jpg')
    graph_count += 1


def part1(seed):
    '''
    Randomly choose 10 images form first 5000 images in training set for each
    digit (Total 100). Put all images on a single image.
    '''
    print "\nPart 1"
    np.random.seed(seed)
    
    # For all 10 digits.
    for i in range (10):
        print "\nChoose 10 for digit " + str(i)
        chosen = random.sample(range(5000),10) 
        
        # Choose 10 images randomlly.
        for j in range(10):
            print chosen[j]
            subplot(10,10,i+j*10+1)
            imshow(M["train"+str(i)][chosen[j]].reshape((28,28)), cmap=cm.gray)
            axis('off')
    
    savefigure()
    show()
    

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def part2(x, W, b):
    '''
    A function computing network. Inputs have dimensions(n, m):
    x: 784 x 1 (matrix, or list of lists)
    W: 784 x 10 (matrix, or list of lists)
    b: 10 x 1 (matrix, or list of lists)
    And output has dimension 10 x 1 (matrix). To ensure matrix multiplication is 
    applicable, the actual calculation is (x dot W)^T + b.
    '''
    return dot(matrix(W).T, matrix(x)) + b
    
    
def cost(y, y_):
    return -sum(y_*log(y)) 
    
    
def combineWb(W, b):
    '''
    Return a matrix whose first column is b and the rest is W.
    [b] + [  W  ] -> [   Wb   ]
    '''
    return vstack((b.T,W))


def splitWb(Wb):
    '''
    Return two matrices. Second one (b) contains only the first column of the input 
    matrix, and the first one (W) contain the rest.
    [   Wb   ] -> [b], [  W  ]
    '''
    return matrix(Wb[1:]), matrix(Wb[0].T)
    

def part3(y, x, Wb):
    '''
    Take list of actual y indices, input x_vector, matrix W and list b.
    Return list of gradients of the total cost with respect to W and b.
    '''
    deriv_sum = zeros(shape(Wb))
    for i in range(len(x)):
        x1 = vstack((matrix([1]), x[i]))
        W, b = splitWb(Wb)
        y_hot = [1 if ys == y[i] else 0 for ys in range(shape(Wb)[1])]
        deriv_sum += x1 * (softmax(part2(x[i], W, b).tolist()).T - matrix(y_hot))
    return deriv_sum

    
def part4(seed):
    '''
    Randomly generate W and b, choose some varification coordinates in Wb, pass 
    both W and b to part3 and finite-difference approximation on the cost function. 
    Print the results.
    '''
    random.seed(seed)
    b = matrix(np.random.random_integers(1, 1, (10, 1)))/100.0
    W = matrix(np.random.random_integers(1, 2, (784, 10)))/100.0
    Wb = combineWb(W, b)
    
    # Choose input x and set corresponding y
    answer, x, y, y_hot = generate_xy(5)  # Sample size 5
    
    # Initializations
    h = 0.0001
    df = part3(y, x, Wb)
    print "Part 4\nCalculated  Estimated   at coordinate (x, y) of W\n"
    
    # Compare several coordinates in W
    for i in range(10): # Compare 10 coordinates in W
    
        # x and y coordinate for comparison.
        checkx = answer
        checky = random.randint(1,784)
        
        calculated = df.tolist()[checky][checkx]  # Shape 785 x 10
        
        # Add h to W_alt.
        Wb_alt= Wb.tolist()  # Shape 785 x 10
        Wb_alt[checky][checkx] += h
        W_alt, b_alt = splitWb(matrix(Wb_alt))
        
        estimated = 0
        for k in range(len(x)):
            estimated += (cost(softmax(part2(x[k], W_alt, b_alt)),y_hot) - cost(softmax(part2(x[k], W, b)),y_hot))/h
        
        print calculated, estimated, "at coordinate (", checkx, ", ", checky, ")"


def mini_batch(initWb, rate, amount, size, initWb0=None):
    '''
    Run mini-batch on training set, tune W and b for amount many of times at a 
    rate of rate. Apply W and b on test set for performance data.
    Return well tuned Wb, and list of success rate on test set. If second Wb,
    (initWb0) is given, tune both for the two layer network.
    '''
    counter = 1
    cost_r, cost_l = [0], [0]
    rate_train, rate_test = [], []
    Wb = initWb.copy()  # Just to be save?
    W, b = splitWb(Wb)
    if initWb0 is not None:
        Wb0 = initWb0.copy()
        W0, b0 = splitWb(Wb0)
    else:
        Wb0 = None

    # Construct test set and their answers once and for all.
    x_test, x_trainfull, y_test, y_trainfull = [], [], [], []
    for cases in range(10):  # 10 digits
        ranges_test = range(len(M["test"+str(cases)]))
        for t in ranges_test:
            x_test.append(matrix(M["test"+str(cases)][t]/255.0).T)
            y_test.append(cases)
        ranges_trainfull = range(len(M["train"+str(cases)]))
        for t in ranges_trainfull:
            x_trainfull.append(matrix(M["train"+str(cases)][t]/255.0).T)
            y_trainfull.append(cases)

    for iteration in range(amount): 

        # Construct training set and their answers. Choose size many randomly.
        x_train, y_train = [], []
        for cases in range(10):  # 10 digits
            ranges_train = random.sample([p for p in range(5000)], size)
            for s in ranges_train:
                x_train.append(matrix(M["train"+str(cases)][s]/255.0).T)
                y_train.append(cases)

        # Step. Tune W and b.
        print "\nIteration "+str(counter)
        counter += 1
        if initWb0 is not None:
            step = part7(y_train, x_train, W, b, W0, b0)
            Wb0 -= rate* step[1]
            W0, b0 = splitWb(Wb0)
            Wb -= rate*step[0]
        else:
            Wb -= rate*part3(y_train, x_train, Wb)
        W, b = splitWb(Wb)


        # Test training set accuracy and add to success_rate.
        success_count = 0
        for i in range(len(x_trainfull)):
            if initWb0 is not None:
                temp = -log(forward(x_trainfull[i], W0, b0, W, b)[2].tolist()[y_trainfull[i]])[0]
                if classify(x_trainfull[i], W, b, W0, b0) == y_trainfull[i]:
                    success_count += 1
            elif classify(x_trainfull[i], W, b) == y_trainfull[i]:
                temp = -log(part2(x_trainfull[i], W, b).tolist()[y_trainfull[i]])[0]
                success_count += 1

            cost_r[-1] += temp if not isnan(temp) else 0

        rate_train.append(success_count*100/float(len(x_trainfull)))  # times 100%
        print "Training set Accuracy: "+str(rate_train[-1]) + "%"

        # Test test set accuracy and add to success_rate.
        success_count = 0
        for i in range(len(x_test)):
            if initWb0 is not None:
                temp =  -log(forward(x_trainfull[i], W0, b0, W, b)[2].tolist()[y_trainfull[i]])[0]
                if classify(x_test[i], W, b, W0, b0) == y_test[i]:
                    success_count += 1
            elif classify(x_test[i], W, b) == y_test[i]:
                temp =  -log(part2(x_trainfull[i], W, b).tolist()[y_trainfull[i]])[0]
                success_count += 1
            cost_l[-1] += temp if not isnan(temp) else 0

        rate_test.append(success_count*100/float(len(x_test)))  # times 100%
        print "Test set Accuracy: "+str(rate_test[-1])+"%"

        # Record the cost.
        print "Cost of training set: "+str(cost_r[-1])
        print "Cost of test set: "+str(cost_l[-1])
        cost_r.append(0)
        cost_l.append(0)

    return Wb, Wb0, rate_train, rate_test, cost_r[:-1], cost_l[:-1]
    
    
def classify(x, W1, b1, W0=None, b0=None):
    '''
    Return the digit the network classifies.
    Softmax is omitted since scale does not affect classification.
    '''
    if W0 is not None and b0 is not None:
        guess = forward(x, W0, b0, W1, b1)[1].tolist()
    else:
        guess = part2(x, W1, b1).tolist()
    return guess.index(max(guess))


def part5(rate, seed, amount, size):
    '''
    Perform mini-batch on subset of training set and return a tuned Wb. Pass the 
    training subset and test subset along with arguments to mini-batch. Plot a 
    line graph on the accuracy and number of iterations.
    '''
    random.seed(seed)
    print "\nPart 5\nMini-batch with iteration=", amount, " rate=", rate, "size=", size, "\n"
    
    init_b = zeros((10, 1))
    init_W = matrix(np.random.random_integers(1, 2, (784, 10)))/100.0
    init_Wb= combineWb(init_W, init_b)

    # Mini-batch function call, no second layer, temp is thrown away.
    Wb, temp, rate_train, rate_test, cost_r, cost_l = mini_batch(init_Wb, rate, amount, size)
    W, b = splitWb(Wb)
    
    # Plot the line charts and save to file.
    plot(range(1,amount+1), rate_train)
    plot(range(1,amount+1), rate_test)
    xlabel('Iterations')
    ylabel('Success Rate %')
    legend(["Training set","Test set"], loc='upper left')
    savefigure()
    show()

    print shape(cost_l)
    plot(range(1,amount+1), cost_r)
    plot(range(1,amount+1), cost_l)
    xlabel('Iterations')
    ylabel('Cost')
    legend(["Training set","Test set"], loc='lower left')
    savefigure()
    show()

    # Find 20 successes and 10 failures.
    c_success, c_failure = 0, 0
    while c_success < 21 or c_failure < 11:
        
        digit = random.randint(0,9)
        sample_i = random.randint(0, len(M["test"+str(digit)]))
        x = matrix(M["test"+str(digit)][sample_i]).T
        
        if digit == classify(x, W, b):
            c_success += 1
            if c_success >= 21:
                continue
            subplot(7, 5, c_success)
        else:
            c_failure += 1
            if c_failure >= 11:
                continue
            subplot(7, 5, 25+c_failure)
        imshow(x.reshape(28, 28), cmap=cm.gray)
        axis('off')
    savefigure()
    show()
    return Wb


def part6(Wb):
    '''
    Shows the heat maps of the tuned weights.
    '''
    Wb = Wb.T.tolist()
    figure(figsize = (10,20))  # Image size

    for i in range(len(Wb)):
        subplot(5,2,i+1)
        imshow(array(Wb[i][1:]).reshape((28, 28)), cmap = cm.coolwarm)
        axis('off')
        cb=colorbar(aspect=5)
        cb.ax.tick_params(labelsize=9)  # Font size

    savefigure()
    show()


def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    #L1 = tanh_layer(L0, W1, b1)
    L1 = dot(W1.T, L0) + b1  # if you don't want tanh at the top layer
    output = softmax(L1)
    return L0, L1, output


def dcdw0(y, h, x, W1, b1, W0, b0):
    '''
    Compute the gradient of C over W0.
    '''
    deriv_sum = matrix(zeros(shape(combineWb(W0, b0))))

    for k in range(len(x)):  # for every input images
        x1 = vstack((matrix([1]), x[k]))
        y_hot = [1 if ys == y[k] else 0 for ys in range(shape(b1)[0])]

        prefix = softmax(part2(h[k], W1, b1).tolist()).T - matrix(y_hot)  # Shape 1 x 10
        sech2ed = sech2(part2(x[k], W0, b0).T.tolist()[0])  # sech(..), shape 1 x 300
        h_to_cost = dot(prefix, W1.T).tolist()[0]  # Shape 1 x 300
        l_to_h = [sech2ed[i]*h_to_cost[i] for i in range(300)]  # Shape 1 x 300
        deriv_sum += dot(x1, matrix(l_to_h))  # Shape 785 x 300
    return deriv_sum


def sech2(list):
    return [1.0/cosh(l)**2 for l in list]


def part7(y, x, W1, b1, W0, b0):
    '''
    x,       W1,       b1,     W0,        b0       has shape
    784 x 1, 300 x 10, 10 x 1, 784 x 300, 300 x 1  respectively.
    Compute the derivative with respect to the multi-layer network with one tanh layer.
    '''
    h = [tanh_layer(xs, W0, b0) for xs in x]
    Wb1 = combineWb(W1, b1)
    dCdW1 = part3(y, h, Wb1)  # Shape 301 x 10
    dCdW0 = dcdw0(y, h, x, W1, b1, W0, b0)  # Shape 785 x 10

    return dCdW1, dCdW0


def generate_xy(size):
    '''
    For part 4 and 8 generate random x and y of size.
    '''
    answer = random.randint(0, 9)
    ranges = random.randint(0, 5000)
    x = []
    for k in range(4):
        x.append(matrix(M["train"+str(answer)][ranges]/255.0).T) # List of training samples

    y = [answer]*4
    y_hot = [1 if answer == i else 0 for i in range(10)]

    return answer, x, y, y_hot


def part8(seed):
    random.seed(seed)

    b0 = matrix(np.random.random_integers(1, 1, (300, 1)))/100.0
    W0 = matrix(np.random.random_integers(1, 2, (784, 300)))/100.0
    b1 = matrix(np.random.random_integers(1, 1, (10, 1)))/100.0
    W1 = matrix(np.random.random_integers(1, 2, (300, 10)))/100.0

    # Choose input x and set corresponding y
    answer, x, y, y_hot = generate_xy(5)

    # Initializations
    h = 0.00001
    df1, df0 = part7(y, x, W1, b1, W0, b0)
    print "Part 8\nCalculated  Estimated   at coordinate (x, y) of W1\n"

    # Compare several coordinates in Ws
    for i in range(10): # Compare 10 coordinates in W1

        # x and y coordinate for comparison.
        checkx = answer
        checky = random.randint(0,300)

        calculated = df1.tolist()[checky][checkx]  # Shape 301 x 10

        # Add h to W_alt.
        Wb1_alt= combineWb(W1, b1).tolist()
        Wb1_alt[checky][checkx] += h
        W1_alt, b1_alt = splitWb(matrix(Wb1_alt))

        estimated = 0
        for k in range(len(x)):
            estimated += (cost(forward(x[k], W0, b0, W1_alt, b1_alt)[2],y_hot)
                          - cost(forward(x[k], W0, b0, W1, b1)[2],y_hot))/h

        print calculated, estimated, "at coordinate (", checkx, ", ", checky, ")"


    print "\nCalculated  Estimated   at coordinate (x, y) of W0\n"
    for i in range(10): # Compare 10 coordinates in W0

        # x and y coordinate for comparison.
        checkx = random.randint(0, 299)
        checky = random.randint(0, 784)

        calculated = df0.tolist()[checky][checkx]  # Shape 301 x 10

        # Add h to W_alt.
        Wb0_alt= combineWb(W0, b0).tolist()
        Wb0_alt[checky][checkx] += h
        W0_alt, b0_alt = splitWb(matrix(Wb0_alt))

        estimated = 0
        for k in range(len(x)):
            estimated += (cost(forward(x[k], W0_alt, b0_alt, W1, b1)[2],y_hot)
                          - cost(forward(x[k], W0, b0, W1, b1)[2],y_hot))/h

        print calculated, estimated, "at coordinate (", checkx, ", ", checky, ")"


def part9(rate, seed, amount, size):
    '''
    Perform mini-batch on subset of training set and return a tuned Wb. Pass the
    training subset and test subset along with arguments to mini-batch. Plot a
    line graph on the accuracy and number of iterations.
    x,       W1,       b1,     W0,        b0       has shape
    784 x 1, 300 x 10, 10 x 1, 784 x 300, 300 x 1  respectively.
    '''
    random.seed(seed)
    print "\nPart 9\nMini-batch with iteration=", amount, " rate=", rate, "size=", size, "\n"

    init_b = zeros((10, 1))
    init_W = matrix(np.random.random_integers(1, 2, (300, 10)))/500.0
    init_Wb= combineWb(init_W, init_b)
    init_b0 = zeros((300, 1))
    init_W0 = matrix(np.random.random_integers(1, 2, (784, 300)))/500.0
    init_Wb0= combineWb(init_W0, init_b0)

    # Mini-batch function call
    Wb, Wb0, rate_train, rate_test, cost_r, cost_l = mini_batch(init_Wb, rate, amount, size, init_Wb0)
    W, b = splitWb(Wb)
    W0, b0 = splitWb(Wb0)

    # Plot the line charts and save to file.
    plot(range(1,amount+1), rate_train)
    plot(range(1,amount+1), rate_test)
    xlabel('Iterations')
    ylabel('Success Rate %')
    legend(["Training set","Test set"], loc='upper left')
    savefigure()
    show()

    plot(range(1,amount+1), cost_r)
    plot(range(1,amount+1), cost_l)
    xlabel('Iterations')
    ylabel('Cost')
    legend(["Training set","Test set"], loc='lower left')
    savefigure()
    show()

    # Find 20 successes and 10 failures.
    c_success, c_failure = 0, 0
    while c_success < 21 or c_failure < 11:

        digit = random.randint(0,9)
        sample_i = random.randint(0, len(M["test"+str(digit)])-1)
        x = matrix(M["test"+str(digit)][sample_i]).T

        if digit == classify(x, W, b, W0, b0):
            c_success += 1
            if c_success >= 21:
                continue
            subplot(7, 5, c_success)
        else:
            c_failure += 1
            if c_failure >= 11:
                continue
            subplot(7, 5, 25+c_failure)
        imshow(x.reshape(28, 28), cmap=cm.gray)
        axis('off')
    savefigure()
    show()
    return Wb, Wb0


def part10(Wb, seed):
    '''
    Shows the heat maps of the tuned weights.
    '''
    random.seed(seed)
    Wb = Wb.T.tolist()
    figure(figsize = (20,20))  # Image size
    counter = 1

    for i in random.sample(range(0,299), 10):
        plt = subplot(10,10,counter)
        counter += 1
        imshow(array(Wb[i][1:]).reshape((28, 28)), cmap = cm.coolwarm)
        axis('off')
        cb=colorbar(aspect=5)
        cb.ax.tick_params(labelsize=9)  # Font size
        plt.set_title("# "+str(i))

    savefigure()
    show()


if __name__ == '__main__':
    print "\nCSC321 Assignment 2\n"
    part1(3857)
    # Passed

    # Part 2 and 3 are function implementations
    # Passed, Passed
    
    part4(3277)
    # Passed
    
    result = part5(0.001, 39966, 200, 60)
    # Passed
    
    part6(result)
    # Passed

    # Part 7 is function implementation.
    # Passed

    part8(34985)
    # Passed

    result2, result3 = part9(0.001, 9873, 300, 50)
    # Passed

    part10(result3, 23984)
    # Passed

    # Save the results for debug.
    with open(dir +"Wresults.txt", "w") as f:
        cPickle.dump((result, result2, result3), f)
