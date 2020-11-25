# Define customized layers
import tensorflow as tf
from tensorflow import keras

class BatchNormalization(keras.layers.BatchNormalization):
    # def __init__(self, *args, **kwargs):
    #     return super(BatchNormalization, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        # training is a call argument, when model is training, it would be True value and vice versa
        # Official: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
        # BatchNormalize Intro: https://stackoverflow.com/questions/50047653/set-training-false-of-tf-layers-batch-normalization-when-training-will-get-a
        # When training is True, it would update moving average of mean and variance. Thus, we can get a better normalization result
        # If the self.trainable is set to True, it would let training be true
        if not training:
            return super(BatchNormalization, self).call(inputs, training=False, **kwargs)
        else:
            return super(BatchNormalization, self).call(inputs, training=(not self.trainable), **kwargs)
