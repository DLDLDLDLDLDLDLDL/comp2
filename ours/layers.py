# Define customized layers
import tensorflow as tf

class BatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        reutrn super(BatchNormalization, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        # training is a call argument, when model is training, it would be True value and vice versa
        # Here we want to 
        if 
