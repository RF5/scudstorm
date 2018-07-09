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
from common.custom_layers import SampleCategoricalLayer, OneHotLayer
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

## Custom layer mapping
custom_keras_layers = {
    'SampleCategoricalLayer': SampleCategoricalLayer,
    'OneHotLayer': OneHotLayer}
    
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
        with tf.name_scope(str(self.name) + 'Model'):
            self.input = Layers.Input(shape=(int(constants.map_width/2), constants.map_height, input_channels))

            self.base = self.add_base()
            self.get_non_spatial = self.add_non_spatial(self.base)
            self.get_spatial = self.add_spatial(self.base, self.get_non_spatial)

            self.model = tf.keras.models.Model(inputs=self.input, outputs=[self.get_non_spatial, self.get_spatial])
        #self.model.compile()
        if self.debug:
            print(">> SCUD2 >> Total number of parameters: ", self.model.count_params()) # currently 561 711, max 4 000 000 in paper
        #self.model.compile(optimizer='rmsprop') #gives error of 'only tf native optimizers are supported in eager mode'
        self.tau_lineage = []
        return None

    def step(self, inputs, batch_predict=False):
        '''
        Takes a step of the Scud model.
        If batch_predict is set to True, we assume inputs is a batch of all env obs 
        and return an array of the corresponding actions.
        '''

        if batch_predict == True:
            batch_list = []
            for game_state in inputs:
                if type(game_state) == util.ControlObject:
                    continue
                k, self.rows, self.columns = obs_parsing.parse_obs(game_state)
                batch_list.append(k)
            if len(batch_list) == 0:
                return [(0, 0, 3) for _ in range(len(inputs))]
            spatial = tf.stack(batch_list, axis=0)
        else:
            if self.mask_output == True or type(inputs) == util.ControlObject:
                if self.debug:
                    print("scud ", self.name, 'output masked')
                return 0, 0, 3
            k, self.rows, self.columns = obs_parsing.parse_obs(inputs)
            spatial = tf.expand_dims(k, axis=0) # now should have shape (1, 8, 8, 25)

        a0, a1 = self.model.predict(spatial)

        arr = []
        rng = 1
        sep_cnt = 0
        if batch_predict:
            rng = len(inputs)
        
        for i in range(rng):
            if batch_predict:
                if type(inputs[i]) == util.ControlObject:
                    arr.append((0, 0, 3))
                    continue

            building = a0[sep_cnt]
            coords = tf.unravel_index(a1[sep_cnt], [self.rows, self.columns/2])

            x = int(coords[0])
            y = int(coords[1])
            if self.debug:
                log("x, y = " + str(x) + ", " + str(y))
            arr.append((x, y, building))
            sep_cnt += 1

        if batch_predict:
            return arr
        else:
            x, y, building = arr[0]
            return x,y,building
        
    def get_flat_weights(self):
        ## Not actually flat weights atm
        weights = self.model.get_weights()
        return weights

    def set_flat_weights(self, params):
        #weights = []
        #for ind in self.weight_spec:
        #    weights.append(tf.reshape(param_vec, shape=))
        self.model.set_weights(params)

    def add_spatial(self, net, a0):
        '''
        Gets the spatial action of the network
        '''
        if self.debug:
            log("getting spatial action")
            s = Stopwatch()

        one_hot_a0 = OneHotLayer(constants.n_base_actions)(a0)

        k = net.get_shape().as_list()
        broadcast_stats = Layers.RepeatVector(int(k[1]*k[2]))(one_hot_a0)
        broadcast_stats2 = Layers.Reshape((k[1], k[2], constants.n_base_actions))(broadcast_stats)
        net = Layers.concatenate([net, broadcast_stats2], axis=-1) # (?, 8, 8, 38)

        net = Layers.Conv2D(32, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv")(net)
        net = Layers.Conv2D(8, [1, 1],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="reduce1")(net)
        net = Layers.Conv2D(32, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv2")(net)
        net = Layers.Conv2D(16, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv3")(net)
        net = Layers.Conv2D(1, [1, 1],
            strides=1,
            padding='SAME',
            name="conv1x1")(net)

        logits = Layers.Flatten()(net)

        a1_sampled = SampleCategoricalLayer()(logits)

        if self.debug:
            log("Finished spatial action inference. Took: " + s.delta)
        return a1_sampled

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
        a0_logits = Layers.Dense(constants.n_base_actions,
                    name="a0")(non_spatial)

        a0_sampled = SampleCategoricalLayer()(a0_logits)

        if self.debug:
            log("Finished non-spatial action. Took: " + s.delta)

        return a0_sampled

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
        print(">> SCUD >> ", self.name, " saved model to file ", str(path)[-60:])
    
    def load(self, filepath, savename):
        if savename is None:
            path = os.path.join(filepath, str(self.name) + '.h5')
        else:
            if savename.endswith('.h5') == False:
                path = os.path.join(filepath, str(savename) + '.h5')
            else:
                path = os.path.join(filepath, str(savename))
        self.model = tf.keras.models.load_model(path, custom_objects=custom_keras_layers)
        print(">> SCUD >> ", self.name ," had model restored from file ", str(path)[-60:])

    def __str__(self):
        return "SCUD2 [Name: {:20} | Masking: {:3} | Refbot pos: {:2d}]".format(self.name, self.mask_output, self.refbot_position)
    

if __name__ == '__main__':
    k = Stopwatch()
    s = Scud('www', debug=True)
    we = s.get_flat_weights()
    log("Round-time was {}".format(k.delta))
    
