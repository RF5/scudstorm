'''
Useful Keras layers.

They wrap fairly critical base tensorflow functions.

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

def add_inception_resnet_A(net, name_prefix):
    initial = net
    net_1x1a = Layers.Conv2D(32, [1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_A_1x1a")(net)

    net_1x1b = Layers.Conv2D(32, [1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_A_1x1b")(net)
    net_3x3b = Layers.Conv2D(32, [3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_A_3x3b")(net_1x1b)

    net_1x1c = Layers.Conv2D(32, [1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_A_1x1c")(net)
    net_3x3c = Layers.Conv2D(48, [3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_A_3x3c")(net_1x1c)
    net_3x3c2 = Layers.Conv2D(64, [3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_A_3x3c2")(net_3x3c)

    conclat = Layers.concatenate([net_1x1a, net_3x3b, net_3x3c2], axis=-1) # 128 channels

    final_1x1 = Layers.Conv2D(64, [1, 1], strides=1, padding='SAME', activation='linear', name=name_prefix + "inception_A_1x1f")(conclat)
    added = Layers.Add()([final_1x1, initial])
    return added

def add_inception_resnet_B(net, name_prefix):
    initial = net
    net_1x1a = Layers.Conv2D(192, [1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_B_1x1a")(net)

    net_1x1b = Layers.Conv2D(128, [1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_B_1x1b")(net)
    net_8x1b = Layers.Conv2D(128, [8, 1], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_B_8x1b")(net_1x1b)
    net_1x8b = Layers.Conv2D(128, [1, 8], strides=1, padding='SAME', activation=tf.nn.relu, name=name_prefix + "inception_B_1x8b")(net_8x1b)

    conclat = Layers.concatenate([net_1x1a, net_1x8b], axis=-1) # 128 channels
    final_1x1 = Layers.Conv2D(64, [1, 1], strides=1, padding='SAME', activation='linear', name=name_prefix + "inception_B_1x1f")(conclat)
    added = Layers.Add()([final_1x1, initial])
    return added
