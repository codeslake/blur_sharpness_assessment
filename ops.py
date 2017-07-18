import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum, 
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)
    
def bnorm(x, train, epsilon=1e-5, momentum = 0.9, name = "batch_norm"):
    
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=train,
                                        scope=name)



def instance_norm_wrapper(input, name, is_training, decay = 0.999):
    with tf.variable_scope(name):
        shape = input.get_shape()
        var_shape = [shape[-1]]
        scale = tf.get_variable('scale', var_shape, initializer = tf.constant_initializer(1.0))
        shift = tf.get_variable('shift', var_shape, initializer = tf.constant_initializer(0.0))
        pop_mean = tf.get_variable('pop_mean', var_shape, initializer = tf.constant_initializer(0.0), trainable = False)
        pop_var = tf.get_variable('pop_var', var_shape, initializer = tf.constant_initializer(1.0), trainable = False)
        epsilon = 1e-3
        
        if is_training:
            mu, sigma_sq = tf.nn.moments(input, [1,2], keep_dims = True)
            train_mean = tf.assign(pop_mean, decay * pop_mean + (1 - decay) * mu)
            train_var = tf.assign(pop_var, decay * pop_var + (1 - decay) * sigma_sq)
            
            with tf.control_dependencies([train_mean, train_var]):
                return instance_norm_for_wrapper(input, mu, sigma_sq, scale, shift, epsilon)
        else:
            return instance_norm_for_wrapper(input, pop_mean, pop_var, scale, shift, epsilon)
            
    
def instance_norm_for_wrapper(input, pop_mean, pop_var, scale, shift, epsilon):
    normalized = (input - pop_mean)/(pop_var + epsilon)**(.5)
    return scale * normalized + shift

def instance_norm(input, name):
    with tf.variable_scope(name):
        shape = input.get_shape()
        var_shape = [shape[-1]]

        scale = tf.get_variable('scale', var_shape, initializer = tf.constant_initializer(1.0))
        shift = tf.get_variable('shift', var_shape, initializer = tf.constant_initializer(0.0))
        epsilon = 1e-3

        mu, sigma_sq = tf.nn.moments(input, [1,2], keep_dims = True)

        normalized = (input - mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift
    
def minibatch_discrimination(input_, output_dim, dim_per_kernel = 5, name = 'minibatch_discrim'):
    batch_size, input_h, input_w, input_dim = input_.get_shape().as_list()[0:4]
    
    with tf.variable_scope(name):
        activation = conv2d(input_, output_dim * dim_per_kernel,
                            k_h = input_h, k_w = input_w, padding = 'VALID',
                            name = 'minibatch_discrim_fconv', return_tuple = True)
        
        activation = tf.reshape(activation, [batch_size, output_dim, dim_per_kernel])
        
        tmp1 = tf.expand_dims(activation, 3)
        tmp2 = tf.transpose(activation, perm=[1,2,0])
        tmp2 = tf.expand_dims(tmp2, 0)
        abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
        f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])

        b = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        f = f + b
        
        return tf.reshape(f, [batch_size, 1, 1, output_dim])

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(axis = 3, values=[x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h = 3, k_w = 3, d_h = 1, d_w = 1, stddev=0.02, padding = 'SAME', name = 'conv2d', return_tuple = False):
    with tf.variable_scope(name):
        with tf.variable_scope('weights'):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                    initializer=tf.contrib.layers.variance_scaling_initializer())
            variable_summaries(w)
        activation = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        
        if return_tuple:
            return activation
            
        with tf.variable_scope('biases'):
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            variable_summaries(biases)
        activation = tf.reshape(tf.nn.bias_add(activation, biases), activation.get_shape())
    
        return activation

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        '''
        #conv2d_transpose
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                            strides=[1, d_h, d_w, 1], padding='VALID')


        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        '''
        # resize deconv
        output = tf.image.resize_nearest_neighbor(input_, [output_shape[1], output_shape[2]])
        output = conv2d(output, output_shape[3], name = 'resize_conv')

        return output 

def total_variation_regularization(input):
    shape = input.get_shape().as_list()
    hor_mask = np.ones((1, 2, 3, 1), dtype = np.float32)
    hor_mask[0, 1, :, :] = -1
    w = tf.ones([1, 2, 3, 1])
    w *= tf.constant(hor_mask, dtype = tf.float32)
    horiz = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID')

    ver_mask = np.ones((2, 1, 3, 1), dtype = np.float32)
    ver_mask[1, 0, :, :] = -1
    w = tf.ones([2, 1, 3, 1])
    w *= tf.constant(hor_mask, dtype = tf.float32)
    vert = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID')

    return tf.reduce_sum(tf.square(horiz)) + tf.reduce_sum(tf.square(vert))

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def get_gram_mtx(self, features_list, shape):
    features_list_a = tf.transpose(features_list, [3, 1, 0, 2])#[batch, width, feature#, height]
    features_list_b = tf.transpose(features_list, [3, 1, 2, 0])#[batch, width, height, feature#]

    texture_feature = tf.matmul(features_list_a, features_list_b) #[batch, width, feature# feature#]
    texture_feature = tf.transpose(texture_feature, [0, 2, 3, 1]) #[batch, feature#, feature#, width]
    texture_feature = tf.reduce_sum(texture_feature, 3) #[batch, feature#, feature#]
    
    size = shape[1] * shape[2] * shape[3]
    
    return texture_feature / size

def KL_loss(mu, log_sigma):
    with tf.variable_scope('KL_divergence'):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        
        return loss
        
