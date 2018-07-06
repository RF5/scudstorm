'''
Useful Keras layers.

They wrap fairly critical basae tensorflow functions.

Author: Matthew Baas
'''

import tensorflow as tf
Layers = tf.keras.layers

class SampleCategoricalLayer(Layers.Layer):
    def __init__(self, **kwargs):
        super(SampleCategoricalLayer, self).__init__(**kwargs)

    def build(self, in_shape):
        super(SampleCategoricalLayer, self).build(in_shape)

    def call(self, x):
        dist = tf.distributions.Categorical(logits=x)
        return dist.sample()
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], )

class OneHotLayer(Layers.Layer):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(OneHotLayer, self).__init__(**kwargs)

    def build(self, in_shape):
        if len(in_shape) != 1:
            raise ValueError("input tensor must be (batches, ) i.e rank 1! But found ", in_shape)
        super(OneHotLayer, self).build(in_shape)

    def call(self, x):
        meme = tf.keras.backend.one_hot(x, num_classes=self.num_classes)
        return meme
    
    def compute_output_shape(self, input_shape):
        outshape = tf.TensorShape([input_shape[0], tf.Dimension(self.num_classes),])
        return outshape