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

    def get_config(self):
        base_config = super(SampleCategoricalLayer, self).get_config() # eg {'name': 'one_hot_layer_2', 'trainable': True, 'dtype': 'float32'}
        # custom layer variables to save go here
        return base_config

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

    def get_config(self):
        base_config = super(OneHotLayer, self).get_config() # eg {'name': 'one_hot_layer_2', 'trainable': True, 'dtype': 'float32'}
        # custom layer variables to save go here
        base_config['num_classes'] = self.num_classes
        return base_config