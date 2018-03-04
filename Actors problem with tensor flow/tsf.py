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
import os
from scipy.ndimage import filters
import urllib
import tensorflow as tf
import random
import cPickle


act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco',
'Peri Gilpin', 'Angie Harmon']

# Please modify the following line inorder to run the program. The directory should
# contain folders 'uncropped', 'cropped' and files 'subset_actors.txt',
#'subset_actresses.txt', this file, and the provided codes.
dir = "./"

database = []  # Store all flattened images  1 x 32 x 32
alexbase = []  # Store all images  1 x 127 x 127
answers = []  # Store all one-hot answers  1 x 10
conv4base = []  # Store conv4 features of all images extracted from alexnet  13 x 13 x 384




def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()



def crop_image(filename, line, size, under):
    '''
    Open, crop, resize the image given and turn it into grayscale image. Save it under
    cropped folder.
    '''
    try:
        img = imread(dir+"uncropped/" + filename)
    except IOError as e:
        print "Could not find image"
        return

    # Crop image given box coordinates
    cord = line.split()[5].split(',')
    try:
        i_cropped = img[int(cord[1]):int(cord[3]),int(cord[0]):int(cord[2])]
    except IndexError as e:
        print "Could not crop" + filename
        return False

    # Resize image
    try:
        i_resized = imresize(i_cropped, [size, size])
    except ValueError as e:
        print "Could not resize " + filename
        return False

    # Save image
    try:
        imsave(dir+under + filename, i_resized)
    except ValueError as e:
        print "Unsupported file format"
        return False

    print filename
    return True


def populate(seed):
    '''
    Load all images and their answer to database and answers dicts.
    '''
    random.seed(seed)
    all_files = os.listdir(dir+"cropped/")
    random.shuffle(all_files)

    for candidate in all_files:
        for i in range(len(act)):
            if name_match(candidate, act[i]):

                # Open, drop alpha channel, flatten and normalize image.
                img = imread(dir+"croppede/"+candidate)[:,:,:3].flatten()/255.
                database.append(img)

                # Open, drop alpha channel, and normalize image for alexnet.
                img2 = imread(dir+"croppedl/"+candidate)[:,:,:3]/255.
                alexbase.append(img2)

                one_hot = zeros(6)
                one_hot[i] = 1
                answers.append(one_hot)
                break


def name_match(file_name, act_name):
    '''
    Return true if file name starts with the last name of act_name.
    '''
    name = act_name.split()[1].lower()
    return name == file_name[:len(name)]


def get_test(source):
    return vstack(source[-100:]), vstack(answers[-100:])


def get_train(source):
    return vstack(source), vstack(answers)


def get_validation(source):
    return vstack(source[-200:-100]), vstack(answers[-200:-100])


def get_sub_train(source, seed, amount):
    random.seed(seed)
    selection = np.random.choice(range(len(source)-201), amount, replace=False)

    x_batch, y_batch = [], []
    for i in range(len(source)):
        if i in selection:
            x_batch.append([source[i]])
            y_batch.append(answers[i])
    return vstack(x_batch), vstack(y_batch)


def part0(srcfile):
    '''
    Compare lines in srcfile to actor/actress names in act. If match download
    the original image to folder uncropped. Calls crop_img for pre-processing.
    '''
    print "\nPart 0"

    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dir+srcfile):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], dir+"uncropped/"+filename), {}, 30)
                if crop_image(filename, line, 32, "cropped/") == False :
                    os.remove(dir+"uncropped/"+filename)
                else:
                    crop_image(filename, line, 227, "croppedl/")
                    crop_image(filename, line, 64, "croppede/")

                i += 1


def part1(seed, batch_size, iterations, rate):
    '''
    Train a fully connected network with one hidden layer to classify images.
    Some lines are pulled or adapted from provided file.
    '''
    print '\nPart 1'
    random.seed(seed)

    # Variable initializations
    x = tf.placeholder(tf.float32, [None, 12288])
    y_ = tf.placeholder(tf.float32, [None, 6])

    nhid = 500
    W0 = tf.Variable(tf.random_normal([12288, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


    # Operation declarations
    # Network definition
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)  # The hidden layer
    layer2 = tf.matmul(layer1, W1)+b1
    y = tf.nn.softmax(layer2)

    # Cost function
    #lam = 0.00000
    #decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))#+decay_penalty)

    # Gradient Descent step
    train_step = tf.train.GradientDescentOptimizer(rate).minimize(NLL)

    # Accuracy calculations
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Execute variable initializations
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    test_x, test_y = get_test(database)
    valid_x, valid_y = get_validation(database)
    accuracy_train, accuracy_valid, accuracy_test = [], [], []

    # Iterations
    for i in range(iterations):
        batch_xs, batch_ys = get_sub_train(database, seed+1, batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 1 == 0:
            print '\ni = ' + str(i+1)
            accu_test = sess.run(accuracy, feed_dict={x: test_x, y_: test_y}) *100
            accuracy_test.append(accu_test)
            print "Test:", accu_test, "%"

            accu_valid = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y}) *100
            accuracy_valid.append(accu_valid)
            print "Validation:", accu_valid, "%"

            batch_xs, batch_ys = get_train(database)
            accu_train = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}) *100
            accuracy_train.append(accu_train)
            print "Train:", accu_train, "%"
            #print "Penalty:", sess.run(decay_penalty)

    plot(range(iterations), accuracy_test)
    plot(range(iterations), accuracy_valid)
    plot(range(iterations), accuracy_train)
    legend(["Test Accuracy", "Validation Accuracy", "Training Accuracy"], loc='lower right')
    xlabel('Iterations')
    ylabel('Accuracy %')
    savefig(dir+'1.jpg')
    show()


############################################
# Adapted from code in myalexnet.py file
############################################


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


def alexnet4():
    '''
    Extract conv4 layer of alexnet applied on all images. Store the results into alexbase.
    '''
    print 'Calculating conv4 values...'

    ############################################
    # Construction phase
    net_data = load("bvlc_alexnet.npy").item()
    x = tf.placeholder(tf.float32, [1,227,227,3])

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    ############################################
    # Execution phase
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    ############################################
    # End of code from file
    ############################################


    for i in range(len(alexbase)):
        conv4base.append(sess.run(conv4, feed_dict = {x:[alexbase[i]]}).flatten())




def part2(batch_size, iterations, rate):
    '''
    Run alexnet on images to get their conv4 layer value. Use this value
    as input to a fully-connected network and train this network. Plot the
    performances on both test and training set.
    Some code are taken/adapted from provided file.
    '''
    print '\nPart 2'

    # Load all conv4 values.
    alexnet4()

    # Variable initializations
    x = tf.placeholder(tf.float32, [None, 64896])
    y_ = tf.placeholder(tf.float32, [None, 6])

    nhid = 500
    W0 = tf.Variable(tf.random_normal([64896, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


    # Operation declarations
    # Network definition
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)  # The hidden layer
    layer2 = tf.matmul(layer1, W1)+b1
    y = tf.nn.softmax(layer2)

    # Cost function
    #lam = 0.00000
    #decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))#+decay_penalty)

    # Gradient Descent step
    train_step = tf.train.GradientDescentOptimizer(rate).minimize(NLL)

    # Accuracy calculations
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Execute variable initializations
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    test_x, test_y = get_test(conv4base)
    valid_x, valid_y = get_validation(conv4base)
    accuracy_train, accuracy_valid, accuracy_test = [], [], []

    # Iterations
    for i in range(iterations):
        batch_xs, batch_ys = get_sub_train(conv4base, 9832, batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 1 == 0:
            print '\ni = ' + str(i+1)
            accu_test = sess.run(accuracy, feed_dict={x: test_x, y_: test_y}) *100
            accuracy_test.append(accu_test)
            print "Test:", accu_test, "%"

            accu_valid = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y}) *100
            accuracy_valid.append(accu_valid)
            print "Validation:", accu_valid, "%"

            batch_xs, batch_ys = get_train(conv4base)
            accu_train = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}) *100
            accuracy_train.append(accu_train)
            print "Train:", accu_train, "%"
            #print "Penalty:", sess.run(decay_penalty)

    plot(range(iterations), accuracy_test)
    plot(range(iterations), accuracy_valid)
    plot(range(iterations), accuracy_train)
    legend(["Test Accuracy", "Validation Accuracy", "Training Accuracy"], loc='lower right')
    xlabel('Iterations')
    ylabel('Accuracy %')
    savefig(dir+'2.jpg')
    show()

    W0=sess.run(W0)
    b0=sess.run(b0)
    W1=sess.run(W1)
    b1=sess.run(b1)
    return (W0, b0, W1, b1)



def part3(seed, batch_size, iterations, rate, nhid, which):
    '''
    Train two networks with 300 and 800 hidden layers. Get two feature maps from each of the hidden layers.
    '''
    random.seed(seed)

    # Read from file to save time.
    if os.path.exists(dir+"part3"+str(nhid)+".pkl"):
        w=cPickle.load(open(dir+"part3"+str(nhid)+".pkl"))
    else:
        # Variable initializations
        x = tf.placeholder(tf.float32, [None, 12288])
        y_ = tf.placeholder(tf.float32, [None, 6])

        W0 = tf.Variable(tf.random_normal([12288, nhid], stddev=0.01, seed=seed))
        b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01, seed=seed+1))
        W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01, seed=seed+2))
        b1 = tf.Variable(tf.random_normal([6], stddev=0.01, seed=seed+3))


        # Operation declarations
        # Network definition
        layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)  # The hidden layer
        layer2 = tf.matmul(layer1, W1)+b1
        y = tf.nn.softmax(layer2)

        # Cost function
        lam = 0.0001
        decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
        NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)

        # Gradient Descent step
        train_step = tf.train.GradientDescentOptimizer(rate).minimize(NLL)


        # Execute variable initializations
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        # Iterations
        for i in range(iterations):
            batch_xs, batch_ys = get_sub_train(database, seed+4, batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


        w = sess.run(W0)
        cPickle.dump(package,  open(dir+"part3"+str(nhid)+".pkl", "w"))



    for i in which:
        r=w[0:,i][::3]
        g=w[1:,i][::3]
        b=w[2:,i][::3]
        feature = sum([r,g,b],0).reshape((64,64))
    #feature = sess.run(W0)[:,2].flatten()[::3].reshape((32,32))
        imshow(feature, cmap = cm.coolwarm)
        axis('off')
        cb=colorbar(aspect=5)
        cb.ax.tick_params(labelsize=9)  # Font size
        print i
        show()
        i+=1
    sess.close()
    #savefig(dir+str(nhid)+'.jpg')
    #show()


def part4net(input, package):
    print '\nRunning Alexnet(cov4)+FC...'

    ############################################
    # Construction phase
    net_data = load("bvlc_alexnet.npy").item()
    x = tf.placeholder(tf.float32, [1,227,227,3])

    # W's and b's for the network.
    nhid = 500
    W0 = tf.placeholder(tf.float32, [64896, nhid])
    b0 = tf.placeholder(tf.float32, [nhid])
    W1 = tf.placeholder(tf.float32, [nhid, 6])
    b1 = tf.placeholder(tf.float32, [6])

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    ########################################
    # End of Alexnet, start of Network


    # Operation declarations
    # Network definition
    layer1 = tf.nn.tanh(tf.matmul(tf.reshape(conv4,[1,64896]), W0)+b0)  # The hidden layer
    layer2 = tf.matmul(layer1, W1)+b1
    y = tf.nn.softmax(layer2)
    grad = tf.gradients(y, x)

    ############################################
    # Execution phase

    # Execute variable initializations
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # Find an image of Angie Harmon for classification.
    W0v, b0v, W1v, b1v= package[0], package[1], package[2], package[3]
    dict={x: [input], W0: W0v, b0: b0v, W1: W1v, b1: b1v}
    return sess.run(y, feed_dict=dict), sess.run(grad, feed_dict=dict)



def part4(package):
    print "\nPart 4"
    result = part4net(alexbase[10], package)[0]

    # Find an image of Angie Harmon for classification.
    imshow(alexbase[10])
    savefig(dir+'AngieHarmon.jpg')
    show()

    guess = act[argmax(result)]
    print '\nProbabilities: ',result[0]
    print 'The guess is: ', argmax(result),':',guess
    if argmax(result) == argmax(answers[10]):
        print 'This guess is correct!'
    return guess


def part5(package):
    print "\nPart 5"

    input_x = alexbase[534]
    print "Choosed image of "+act[argmax(answers[534])]

    subplot(1,2,1)
    imshow(input_x)
    axis('off')

    grad = part4net(alexbase[534], package)[1][0]
    shape_grad = shape(grad)
    grad_f = grad.flatten()
    grad_positive = array([g if g>0 else 0 for g in grad_f])  # Remove negative values.
    grad_positive = grad_positive*10e7  # A too simple normalization. Fix if have time.

    subplot(1,2,2)
    imshow(grad_positive.reshape(shape_grad)[0])
    axis('off')
    savefig(dir+'part5.jpg')
    show()







if __name__ == '__main__':
    '''
    Main program, driver of all others and produce all images/results in the report.
    '''
    # The following two lines download images and crop them into two sizes. No guarentee
    # on downloading the exact same images everytime when they are run.
    #part0("subset_actors.txt")
    #part0("subset_actresses.txt")
    # Tested passed

    # Load all images and find their answers.
    print 'Reading images...'
    populate(38745)
    # Working fine

    #part1(298435, 50, 2000, 0.0002)
    # Fixed. Tested passed. 81%

    #package = part2(30, 2000, 0.0002)
    # Still testing. 91%

    if os.path.exists(dir+"package.pkl"):
        package=cPickle.load(open(dir+"package.pkl"))
    else:
        cPickle.dump(package,  open(dir+"package.pkl", "w"))


    print '\nPart 3'
    print 'Training network (300)...'
    part3(309454, 50, 3000, 0.002, 300, range(300))
    part3(876234, 50, 3000, 0.002, 800, range(300))
    # Cannot maintain reproducibility. Results ok. Possible options: feed our W's and b's, or save to pkl and load.

    #part4(package)
    # Working well. Correct guess

    #part5(package)
    # Keep it rolling mate!