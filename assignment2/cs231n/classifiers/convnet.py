import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    [conv-[bn]- relu-pool] X conv_layer_num - [affine-[bn]-relu] X affine_layer_num - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the  convolutional               #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.use_batchnorm = use_batchnorm
        self.num_filters = 32
        self.filter_size = 3
        self.conv_layers_num = 3
        self.affine_layers_num = 3
        self.hidden_layers_num = self.conv_layers_num + self.affine_layers_num
        self.total_layers_num = self.hidden_layers_num + 1

        self.conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        if self.conv_layers_num > 0:
            layer_input_depth = [self.num_filters for i in xrange(self.conv_layers_num)]
            layer_input_depth[0] = C

        for i in xrange(1, self.conv_layers_num + 1):
            self.params['W%d' % i] = np.random.randn(
                self.num_filters, layer_input_depth[i - 1], self.filter_size, self.filter_size) * weight_scale
            self.params['b%d' % i] = np.zeros(self.num_filters)
            if use_batchnorm:
                self.params['gamma%d' % i] = np.ones(self.num_filters)
                self.params['beta%d' % i] = np.zeros(self.num_filters)

        shrink_H = self.pool_param['pool_height']**self.conv_layers_num
        shrink_W = self.pool_param['pool_width']**self.conv_layers_num
        if self.conv_layers_num > 0:
            after_conv_size = np.prod(input_dim[1:]) * \
                self.num_filters / shrink_H / shrink_W
        else:
            after_conv_size = np.prod(input_dim)
        affine_dims = [after_conv_size, 300, 100, 30]

        for i in xrange(1, self.affine_layers_num + 1):
            j = self.conv_layers_num + i
            self.params['W%d' % j] = np.random.randn(
                affine_dims[i - 1], affine_dims[i]) * weight_scale
            self.params['b%d' % j] = np.zeros(affine_dims[i])
            if use_batchnorm:
                self.params['gamma%d' % j] = np.ones(affine_dims[i])
                self.params['beta%d' % j] = np.zeros(affine_dims[i])

        self.params['W%d' % (self.total_layers_num)
                    ] = np.random.randn(affine_dims[self.affine_layers_num], num_classes) * weight_scale
        self.params['b%d' % (self.total_layers_num)] = np.zeros(num_classes)

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in xrange(self.hidden_layers_num)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        cache = []
        o = X
        for i in xrange(1, self.conv_layers_num + 1):
            W = self.params['W%d' % i]
            b = self.params['b%d' % i]
            if self.use_batchnorm:
                gamma = self.params['gamma%d' % i]
                beta = self.params['beta%d' % i]
                o, c = conv_bn_relu_pool_forward(
                    o, W, b, gamma, beta, self.conv_param, self.bn_params[i - 1], self.pool_param)
            else:
                o, c = conv_relu_pool_forward(o, W, b, self.conv_param, self.pool_param)
            cache.append(c)

        for i in xrange(self.conv_layers_num + 1, self.hidden_layers_num + 1):
            W = self.params['W%d' % i]
            b = self.params['b%d' % i]
            if self.use_batchnorm:
                gamma = self.params['gamma%d' % i]
                beta = self.params['beta%d' % i]
                o, c = affine_bn_forward(o, W, b, gamma, beta, self.bn_params[i - 1])
            else:
                o, c = affine_forward(o, W, b)
            cache.append(c)

        o, c = affine_forward(o, self.params['W%d' % self.total_layers_num], self.params[
                              'b%d' % self.total_layers_num])
        cache.append(c)
        scores = o
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the convolutional net,             #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        Ws = ['W%d' % i for i in xrange(1, self.total_layers_num + 1)]
        loss += 0.5 * self.reg * sum(np.sum(self.params[w] ** 2) for w in Ws)

        c = cache.pop()
        dout, dW, db = affine_backward(dout, c)
        grads['W%d' % self.total_layers_num] = dW
        grads['b%d' % self.total_layers_num] = db
        for i in xrange(self.hidden_layers_num, self.conv_layers_num, -1):
            c = cache.pop()
            if self.use_batchnorm:
                dout, dW, db, dgamma, dbeta = affine_bn_backward(dout, c)
                grads['gamma%d' % i] = dgamma
                grads['beta%d' % i] = dbeta
            else:
                dout, dW, db = affine_backward(dout, c)
            grads['W%d' % i] = dW
            grads['b%d' % i] = db

        for i in xrange(self.conv_layers_num, 0, -1):
            c = cache.pop()
            if self.use_batchnorm:
                dout, dW, db, dgamma, dbeta = conv_bn_relu_pool_backward(dout, c)
                grads['gamma%d' % i] = dgamma
                grads['beta%d' % i] = dbeta
            else:
                dout, dW, db = conv_relu_pool_backward(dout, c)
            grads['W%d' % i] = dW
            grads['b%d' % i] = db

        for w in Ws:
            grads[w] += self.params[w] * self.reg
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(bn)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache


def conv_bn_relu_pool_backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
    dx, dw, db = conv_backward_fast(dbn, conv_cache)
    return dx, dw, db, dgamma, dbeta


def affine_bn_forward(x, w, b, gamma, beta, bn_param):
    a, affine_cache = affine_forward(x, w, b)
    out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    cache = (affine_cache, bn_cache)
    return out, cache


def affine_bn_backward(dout, cache):
    affine_cache, bn_cache = cache
    dbn, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    dx, dw, db = affine_backward(dbn, affine_cache)
    return dx, dw, db, dgamma, dbeta


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, affine_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (affine_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    affine_cache, bn_cache, relu_cache = cache
    dr = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(dr, bn_cache)
    dx, dw, db = affine_backward(dbn, affine_cache)
    return dx, dw, db, dgamma, dbeta
