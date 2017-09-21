# Testing script for V35_1
# set each bit of testing data to random 0/1
# The dataset has been divided into 4 subsets
# check the error rate


from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(4321) # for reproducibility?

# specifying the gpu to use
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T

import lasagne
import binary_net
import hdf5storage


if __name__ == "__main__":

    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")

    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))

    print('Loading CIFAR-10 dataset...')

    # read in data in (-1,3*8,32,32) format
    test_set_tp = hdf5storage.loadmat('test_set_x.mat')
    test_set_x = 2* test_set_tp['test_set_x']- 1.
    test_set_x = test_set_x.astype(np.float32)

    # read in labels (targets) in (-1,1) format
    test_set_tp = hdf5storage.loadmat('test_set_y.mat')
    test_set_y = test_set_tp['test_set_y']

    # flatten targets
    test_set_y = np.hstack(test_set_y)
    test_set_y_int = test_set_y.astype(np.int32) - 1 # modify 1-10 to 0-9

    # Onehot the targets
    test_set_y = np.float32(np.eye(10)[test_set_y_int])

    # for hinge loss
    test_set_y = 2* test_set_y - 1.
    test_set_y = test_set_y.astype(np.float32)

#    # Part of test set
    N = 10000
    bit = 0 # (bit+1)th slice
    test_set_x = test_set_x[N:N+10000,:,:,:]
    test_set_x[:,bit:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit:8:24,:,:,].shape)*2-1.
    test_set_x[:,bit+1:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+1:8:24,:,:,].shape)*2-1.
    test_set_x[:,bit+2:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+2:8:24,:,:,].shape)*2-1.
    test_set_x[:,bit+3:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+3:8:24,:,:,].shape)*2-1.
    test_set_x[:,bit+4:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+4:8:24,:,:,].shape)*2-1.
    test_set_x[:,bit+5:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+5:8:24,:,:,].shape)*2-1.
    test_set_x[:,bit+6:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+6:8:24,:,:,].shape)*2-1.
#    test_set_x[:,bit+7:8:24,:,:,] = np.random.randint(2, size=test_set_x[:,bit+7:8:24,:,:,].shape)*2-1.

    test_set_y = test_set_y[N:N+10000,]

    print('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape=(None, 3*8, 32, 32),
            input_var=input)

    # 128C3-128C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=False,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # 256C3-256C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # 512C3-512C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # print(cnn.output_shape)

    # 1024FP-1024FP-10FP
    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], test_err)

    print("Loading the trained parameters and binarizing the weights...")

    # Load parameters
    with np.load('cifar10_parameters.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)

#    # Binarize the weights
#    params = lasagne.layers.get_all_params(cnn)
#    for param in params:
#        # print param.name
#        if param.name == "W":
#            param.set_value(np.float32(2.*np.greater_equal(param.get_value(),0)-1.))

    print('Running...')

    start_time = time.time()

    test_error = val_fn(test_set_x,test_set_y)*100.
    print("test_error = " + str(test_error) + "%")

    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")
