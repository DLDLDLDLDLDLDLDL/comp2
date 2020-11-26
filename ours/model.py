from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import initializers
# from tensorflow.keras import models

# Customized modules
from efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

# Hyperparameters
MOMENTUM = 0.997
EPSILON = 1e-4

# According to EfficientDet paper to set weighted BiFPN depth and depth of heads
# The backbones  of EfficientDet
'''
The corresponding backbone to EfficientDet Phi
B0 -> D0(phi 0), B1 -> D1(phi 1), B2 -> D2(phi 2), B3 -> D3(phi 3), B4 -> D4(phi 4), B5 -> D5(phi 5), B6 -> D6(phi 6)
B6 -> D7(phi 7), B7 -> D7X(phi 8) (IMPORTANT)
The value of phi is corresponding to the order of the following backbone list
'''
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB6, EfficientNetB7]

# The width of BiFPN which is the number of channels, also named 'fpn_num_filters' in efficientdet-tf2/efficientdet.py
# The formular of the paper is W = 64 * (1.35 ^ phi)
w_bifpns = [64, 88, 112, 160, 224, 288, 384,384]

# The depth of BiFPN which is the number of layers, also named 'fpn_cell_repeats' in efficientdet-tf2/efficientdet.py
# The formular of the paper is D = 3 + phi
d_bifpns = [3, 4, 5, 6, 7, 7, 8, 8]

# The input image size of EfficientDet
'''
It is weired that from original paper, the input image size should be following
the input image size of EfficientDet of phi 6, 7, 8(7X) is 1280, 1536, 1536
image_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
'''
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]


depth_heads = [3, 3, 3, 4, 4, 4, 5]

# Copy from original, done
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
''' 
# Build Weighted BiFPN
def build_wBiFPN(features, num_channels, id, freeze_bn=False):

    

def EfficientDet(phi, num_classes = 20, num_anchors = 9):
    assert(phi < 8)
    

    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)

    img_inputs = keras.layers.Input(shape=input_shape)
    x = SeparableConvBlock_c(num_channels=3, kernel_size=3, strides=1)(img_inputs)
    model = keras.models.Model(inputs=[img_inputs], outputs=x, name='efficientdet')

if __name__ == '__main__':
    phi = 0
    EfficientDet(0)
    # x = SeparableConvBlock()