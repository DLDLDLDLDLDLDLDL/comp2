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
weighted_bifpn = [64, 88, 112, 160, 224, 288, 384]
depth_bifpns = [3, 4, 5, 6, 7, 7, 8]
depth_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3]

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

def build_wBiFPN(features, num_channels, id, freeze_bn=False):

    

def EfficientDet(phi, num_classes = 20, num_anchors = 9):
    assert(phi < 8)
    bifpn_num_filters = [64, 88, 112, 160, 224, 288, 384,384]
    bifpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
    backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3]

    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)

    img_inputs = keras.layers.Input(shape=input_shape)
    x = SeparableConvBlock_c(num_channels=3, kernel_size=3, strides=1)(img_inputs)
    model = keras.models.Model(inputs=[img_inputs], outputs=x, name='efficientdet')

if __name__ == '__main__':
    phi = 0
    EfficientDet(0)
    # x = SeparableConvBlock()