# Define customized layers
from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# Customized modules
from efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

# Useless
# Identical to keras.layers.BatchNormalization, but add control to the momentum and epsilon trainable or not
# class BatchNormalization(keras.layers.BatchNormalization):
#     def __init__(self, *args, **kwargs):
#         super(BatchNormalization, self).__init__(*args, **kwargs)

#     def call(self, inputs, training=None, **kwargs):
#         # training is a call argument, when model is training, it would be True value and vice versa
#         # Official: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
#         # BatchNormalize Intro: https://stackoverflow.com/questions/50047653/set-training-false-of-tf-layers-batch-normalization-when-training-will-get-a
#         # When training is True, it would update moving average of mean and variance. Thus, we can get a better normalization result
#         # If the self.trainable is set to True, it would let training be true
#         if not training:
#             return super(BatchNormalization, self).call(inputs, training=False, **kwargs)
#         else:
#             return super(BatchNormalization, self).call(inputs, training=(not self.trainable), **kwargs)

# Implement "Fast Normalized Fusion", done
class wBiFPNAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        input_dim = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=input_dim,
                                initializer=keras.initializers.Constant(1 / input_dim),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.nn.relu(self.w)
        sum = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = sum / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config().update({'epsilon': self.epsilon})
        return config

# Hyperparameters
MOMENTUM = 0.997
EPSILON = 1e-4

# According to EfficientDet paper to set weighted BiFPN depth and depth of heads
# The backbones  of EfficientDet, done
'''
The corresponding backbone to EfficientDet Phi
B0 -> D0(phi 0), B1 -> D1(phi 1), B2 -> D2(phi 2), B3 -> D3(phi 3), B4 -> D4(phi 4), B5 -> D5(phi 5), B6 -> D6(phi 6)
B6 -> D7(phi 7), B7 -> D7X(phi 8) (IMPORTANT)
The value of phi is corresponding to the order of the following backbone list
'''
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB6, EfficientNetB7]

# The width of BiFPN which is the number of channels, also named 'fpn_num_filters' in efficientdet-tf2/efficientdet.py, done
# The formular of the paper is W = 64 * (1.35 ^ phi)
w_bifpns = [64, 88, 112, 160, 224, 288, 384, 384, 384]

# The depth of BiFPN which is the number of layers, also named 'fpn_cell_repeats' in efficientdet-tf2/efficientdet.py, done
# The formular of the paper is D = 3 + phi
d_bifpns = [3, 4, 5, 6, 7, 7, 8, 8, 8]

# The input image size of EfficientDet, done
'''
It is weired that from original paper, the input image size should be following
the input image size of EfficientDet of phi 6, 7, 8(7X) is 1280, 1536, 1536
image_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
'''
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1408]

# The layers of BoxNet & ClassNet
depth_heads = [3, 3, 3, 4, 4, 4, 5, 5, 5]

# Reproduce original, done
def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, name=f'{name}/conv2d')
    f2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    # conv_bn = lambda *args, **kwargs: f2(f1(args, **kwargs))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

# Rewrite SeparableConvBlock with layer subclass, done
class SeparableConvBlock_c(keras.layers.Layer):
    def __init__(self, num_channels, kernel_size, strides, momentum=MOMENTUM, epsilon=EPSILON):
        super(SeparableConvBlock_c, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        # self.name = name
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        self.f1 = keras.layers.SeparableConv2D(
            self.num_channels, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same', use_bias=True, 
            name=f'{self.name}/SeparableConvBlock_c-conv2d')

        self.f2 = keras.layers.BatchNormalization(
            momentum=self.momentum, 
            epsilon=self.epsilon, 
            name=f'{self.name}/SeparableConvBlock_c-bn')
    
    def call(self, inputs):
        return self.f2(self.f1(inputs))

'''
According to the paper, for each BiFPN block, it use depthwise separable convolution for
feature fusion, and add batch normalization and activation after each convolution.
Process:
Depthwise separable convolution -> batch normalization -> activation
''' 
# Build Weighted BiFPN, done
def build_wBiFPN(features, num_channels, id):
    if id == 0:
        '''
        For the first layer of BiFPN, we can only use P3, P4, P5 as inputs
        Don't know why, but many implementation do the same thing
        Depthwise separable convolution -> batch normalization
        '''
        pre = 'pre_'
        bifpn = 'bifpn-'
        _, _, C3, C4, C5 = features

        pn = 3
        P3_in = C3
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P3_in)
        P3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'{bifpn}{id}/{pn}/{pre}bn')(P3_in)
        pn = 4
        P4_in = C4
        P4_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pre}conv2d')(P4_in)
        P4_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn')(P4_in)
        # P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
        #                         name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P4_in)
        # P4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
        #                                     name=f'{bifpn}{id}/{pn}/{pre}bn')(P4_in_2)
        
        pn = 5
        P5_in = C5
        P5_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P5_in)
        P5_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn')(P5_in)
        # P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
        #                         name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P5_in)
        # P5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
        #                                     name=f'{bifpn}{id}/{pn}/{pre}bn')(P5_in_2)

        pn = 6
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='{bifpn}{id}/{pn}/{pre}conv2d')(C5)
        P6_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='{bifpn}{id}/{pn}/{pre}bn')(P6_in)
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='{bifpn}{id}/{pn}/{pre}maxpool')(P6_in)

        pn = 7
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features

    # Top-Down & Bottom-Up BiFPN
    '''
    Px_Up: Top-down(upsampling implementation) to resize the feature map
    Px_Down: Bottom-Up convolution
    Px_td: The intermediate nodes of the layer
    Px_out: The output nodes of the layer

    illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
    refer by zylo117/Yet-Another-EfficientDet-Pytorch

    The process: 
    Depthwise separable convolution -> batch normalization and activation
    '''
    pre = 'fpn_top_down'
    bifpn = 'bifpn-'

    # Top-Down
    # Top-down upsampling
    P7_Up = layers.UpSampling2D()(P7_in)
    # Do Fast Normalized Fusion, then we get the intermediate nodes
    P6_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P7_Up])
    '''
    Although the paper says "add batch normalization and activation after each convolution.", 
    in most of the implementation, they always apply swish before convolution
    '''
    # The original code for swish
    # P6_td = layers.Activation(tf.nn.swish)(P6_td)
    P6_td = tf.nn.swish(P6_td)
    # Separable Convolution and Batch Normalization
    P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_td)
    
    P6_Up = layers.UpSampling2D()(P6_td)
    P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_1, P6_Up])
    # P5_td = layers.Activation(tf.nn.swish)(P5_td)
    P5_td = tf.nn.swish(P5_td)
    P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_td)

    P5_Up = layers.UpSampling2D()(P5_td)
    P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_1, P5_Up])
    # P4_td = layers.Activation(tf.nn.swish)(P4_td)
    P4_td = tf.nn.swish(P4_td)
    P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_td)
    
    P4_Up = layers.UpSampling2D()(P4_td) 
    P3_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P3_in, P4_Up])
    # P3_out = layers.Activation(tf.nn.swish)(P3_out)
    P3_out = tf.nn.swish(P3_out)
    P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P3_out)

    # Bottom-Up
    pre = 'fpn_bottom_up'

    # Bottom-Up pooling
    P3_Down = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
    P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_2, P4_td, P3_Down])
    # The original code for swish
    # P4_out = layers.Activation(tf.nn.swish)(P4_out)
    P4_out = tf.nn.swish(P4_out)
    P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_out)

    P4_Down = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
    P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_2, P5_td, P4_Down])
    # P5_out = layers.Activation(tf.nn.swish)(P5_out) 
    P5_out = tf.nn.swish(P5_out)
    P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_out)

    P5_Down = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
    P6_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P6_td, P5_Down])
    # P6_out = layers.Activation(tf.nn.swish)(P6_out)
    P6_out = tf.nn.swish(P6_out)
    P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_out)

    P6_Down = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
    P7_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P7_in, P6_Down])
    # P7_out = layers.Activation(tf.nn.swish)(P7_out)
    P7_out = tf.nn.swish(P7_out)
    P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P7_out)
    
    return [P3_out, P4_out, P5_out, P6_out, P7_out]