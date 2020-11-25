from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializer
from tensorflow.keras import models

# Customized modules
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3


# According to EfficientDet paper to set weighted BiFPN depth and depth of heads
weighted_bifpn = [64, 88, 112, 160, 224, 288, 384]
depth_bifpns = [3, 4, 5, 6, 7, 7, 8]
depth_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3]

# Hyperparameters
momentum = 0.997
epsilon = 1e-4

