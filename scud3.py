# -*- coding: utf-8 -*-
'''
Scudstorm

Entelect Challenge 2018
Author: Matthew Baas
'''

import json
import os
import random
import common.metrics
import tensorflow as tf
from common.metrics import log, Stopwatch
import numpy as np
import common.util as util
import constants
from common import obs_parsing

Layers = tf.keras.layers
# Sequential = tf.keras.layers.Sequential
# Dense = tf.keras.layers.Dense
# Input = tf.keras.layers.Input

'''
Internal agent config
'''
debug_verbose = False
endpoints = {}
device = 'cpu'
tf.enable_eager_execution()
input_channels = 28
# let an example map size be 20x40, so each player's building area is 20x20
    
class Scud(object):
    
    def __init__(self, name, debug=False):
        '''
        Build agent graph.
        '''

        self.name = name
        self.debug = debug
        self.fitness_score = 0
        self.fitness_averaging_list = []
        self.refbot_position = -1
        self.mask_output = False
        if self.debug:
            log("Running conv2d on " + device)
        # with tf.device('/' + device + ':0'):
        self.input = Layers.Input(shape=(int(constants.map_width/2), constants.map_height, input_channels))
        self.latent_input = Layers.Input(shape=(int(constants.map_width/2), constants.map_height, constants.n_base_actions))

        self.base = self.add_base()
        
        self.get_non_spatial = self.add_non_spatial(self.base)
        self.get_spatial = self.add_spatial(self.base, self.get_non_spatial)

        self.model = tf.keras.models.Model(inputs=self.input, outputs=[self.get_non_spatial, self.base])
        self.latent_model = tf.keras.models.Model(inputs=[self.input, self.latent_input], outputs=self.get_spatial)
        #self.model.compile()
        if self.debug:
            print(">> SCUD2 >> Total number of parameters: ", self.model.count_params()) # currently 561 711, max 4 000 000 in paper
        #self.model.compile(optimizer='rmsprop') #gives error of 'only tf native optimizers are supported in eager mode'
        self.tau_lineage = []
        return None

    def step(self, inputs):
        if self.mask_output == True or type(inputs) == util.ControlObject:
            if self.debug:
                print("scud ", self.name, 'output masked')
            return 0, 0, 3

        k, self.rows, self.columns = obs_parsing.parse_obs(inputs)
        self.spatial = tf.expand_dims(k, axis=0) # now should have shape (1, 8, 8, 25)
        return self.generate_action()
        
    def get_flat_weights(self):
        ## Not actually flat weights atm
        weights = self.model.get_weights()

        return weights

    def set_flat_weights(self, params):
        #weights = []
        #for ind in self.weight_spec:
        #    weights.append(tf.reshape(param_vec, shape=))
        self.model.set_weights(params)

    def generate_action(self):
        '''
        Scud model estimator
        '''

        a0 = self.model.predict(self.spatial)
        #probs = tf.nn.softmax(a0)
        #sample = np.random.choice([0, 1, 2, 3], p=probs)
        print("a0 = ", a0)

        dist = tf.distributions.Categorical(logits=a0)
        sample = dist.sample()
        if self.debug:
            print("a0 = ", a0, a0.shape)
            print(sample)
        building = int(sample) # now an int between 0 and 3
        if self.debug:
            log("a0 = " + str(a0))

        oh = tf.one_hot(indices=building, depth=constants.n_base_actions, axis=-1, name="a0")
        #print(oh)
        intermediary = tf.tile(tf.expand_dims(tf.expand_dims(oh, axis=0), axis=0), [constants.map_height, int(constants.map_width/2), 1])
        a1 = self.latent_model.predict([self.spatial, tf.expand_dims(intermediary, axis=0)])

        print("Final output = ", a1)

        dist2 = tf.distributions.Categorical(logits=a1)
        sample2 = dist2.sample()

        coords = tf.unravel_index(sample2, [self.rows, self.columns/2])
        x = int(coords[0])
        y = int(coords[1])
        if self.debug:
            log("x, y = " + str(x) + ", " + str(y))

        ## loading the state (for RNN stuffs)
        # if self.debug:
        #     log("Loading state")
        #     sss = Stopwatch()
        # _ = np.load('scudstate.npy') # takes ~ 0.031s
        # if self.debug:
        #     log("State loaded. Took: " + sss.delta)

        # ## saving the state (for RNN stuffs)
        # if self.debug:
        #     log("Saving state")
        #     ss = Stopwatch()
        # new_state = net
        # np.save('scudstate.npy', new_state)
        # if self.debug:
        #     log("State saved. Took: " + ss.delta)
        
        #util.write_action(x,y,building)
        return x,y,building

    def add_spatial(self, net, non_spatial_logits):
        '''
        Gets the spatial action of the network
        '''
        if self.debug:
            log("getting spatial action")
            s = Stopwatch()

        print("LATENT SHIT ", self.latent_input)

        k = net.get_shape().as_list()
        broadcast_stats = Layers.RepeatVector(int(k[1]*k[2]))(non_spatial_logits)
        broadcast_stats2 = Layers.Reshape((k[1], k[2], constants.n_base_actions))(broadcast_stats)
        net = Layers.concatenate([net, broadcast_stats2], axis=-1)

        net = Layers.Conv2D(32, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv")(net)
        net = Layers.Conv2D(32, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv2")(net)
        net = Layers.Conv2D(1, [1, 1],
            strides=1,
            padding='SAME',
            name="conv1x1")(net)

        logits = Layers.Flatten()(net)

        #probs = Layers.Dense(flat, activation='softmax', name='softmax')(flat)

        ## DOES NOT WORK WITH EAGER. EAGER = NO MIXING NATIVE TF STUFF WITH KERAS STUFF.
        #dist = tf.distributions.Categorical(logits=flat)
        #sample = dist.sample()

        #coords = tf.unravel_index(sample, [self.rows, self.columns/2])

        if self.debug:
            log("Finished spatial action inference. Took: " + s.delta)
        return logits

    def add_non_spatial(self, net):
        '''
        Infers the non-spatial action of the network
        '''
        if self.debug:
            log("Getting non-spatial action")
            s = Stopwatch()
        flatten = Layers.Flatten()(net)
        non_spatial = Layers.Dense(256,
                    activation=tf.nn.relu,
                    name="non_spatial")(flatten)
        a0 = Layers.Dense(constants.n_base_actions,
                    name="a0")(non_spatial)

        # TODO: possibly softmax this and then transform it into an int from 0 - 4
        # possibly use tf autoregressive distribution
        ## Do not work with eager

        #dist = self.track_layer(tf.distributions.Categorical(logits=a0))
        #sample = self.track_layer(dist.sample())
        #print("ADDING DIST LAYER")
        #a00 = DistLayer()(a0)

        if self.debug:
            log("Finished non-spatial action. Took: " + s.delta)

        return a0

    def add_base(self):
        if self.debug:
            log("Adding base")
            s = Stopwatch()
        with tf.name_scope("adding_base") as scope:
            net = self.input
            for i in range(3):
                net = (Layers.Conv2D(32, [3, 3],
                            strides=1,
                            padding='SAME',
                            activation=tf.nn.relu,
                            name="conv" + str(i)))(net) # ok well this takes 5 seconds

        if self.debug:
            log("Finished adding base. Took: " + s.delta)

        return net

    def squash_fitness_scores(self):
        self.fitness_score = np.mean(self.fitness_averaging_list)
        self.fitness_averaging_list.clear()

    def save(self, filepath, savename=None):  
        if savename is None:
            path = os.path.join(filepath, str(self.name) + '.h5')
        else:
            if savename.endswith('.h5') == False:
                path = os.path.join(filepath, str(savename) + '.h5')
            else:
                path = os.path.join(filepath, str(savename))
        os.makedirs(filepath, exist_ok=True)
        self.model.save(path, include_optimizer=False)
        print(">> SCUD >> ", self.name, " saved model to file ", path)
    
    def load(self, filepath, savename):
        if savename is None:
            path = os.path.join(filepath, str(self.name) + '.h5')
        else:
            if savename.endswith('.h5') == False:
                path = os.path.join(filepath, str(savename) + '.h5')
            else:
                path = os.path.join(filepath, str(savename))
        self.model = tf.keras.models.load_model(path)
        print(">> SCUD >> ", self.name ," had model restored from file ", path)

# class DistLayer(tf.keras.layers.Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         print('---------> inited')
#         super(DistLayer, self).__init__(**kwargs)

#     def build(self, in_shape):
#         print('-------------> built')
#         super(DistLayer, self).build(in_shape)

#     def call(self, x):
#         dist = tf.distributions.Categorical(logits=x)
#         sample = dist.sample()
#         print("SAMEPLE = ", sample)
#         #dist2 = tf.multinomial(logits=x, num_samples=1)
#         #print("MULTINOMIAL SAMPLE = ", dist2)
#         return tf.expand_dims(sample, axis=-1)
    
#     def compute_output_shape(self, input_shape):
#         out = input_shape[0] + tf.Dimension(1)
#         print("CALLED COMPUTE OUTPUT SHAPE! ", out)
#         return (input_shape[0], 1)

if __name__ == '__main__':

    k = Stopwatch()
    s = Scud('www', debug=True)
    we = s.get_flat_weights()
    log("Round-time was {}".format(k.delta))
    
