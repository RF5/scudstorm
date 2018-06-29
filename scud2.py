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
from common import obs_parsing

Layers = tf.keras.layers
# Sequential = tf.keras.layers.Sequential
# Dense = tf.keras.layers.Dense
# Input = tf.keras.layers.Input

'''
Internal agent config
'''
n_base_actions = 4 # number of base actions -- 0=NO OP, 1=DEFENSE, 2=OFFENSE, 3=ENERGY...
debug_verbose = False
endpoints = {}
device = 'gpu'
tf.enable_eager_execution()
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
        with tf.device('/' + device + ':0'):
            self.input = Layers.Input(shape=(4, 4, 25))

            self.base = self.add_base()
            
            self.get_non_spatial = self.add_non_spatial(self.base)
            self.get_spatial = self.add_spatial(self.base)

            self.model = tf.keras.models.Model(inputs=self.input, outputs=[self.get_non_spatial, self.get_spatial])
            #self.model.compile(optimizer=tf.train.AdamOptimizer) #gives error of 'only tf native optimizers are supported in eager mode'
            self.tau_lineage = []
        return None

    def step(self, inputs):
        if self.mask_output == True or type(inputs) == util.ControlObject:
            if self.debug:
                print("scud ", self.name, 'output masked')
            return 0, 0, 3

        k, self.rows, self.columns = obs_parsing.parse_obs(inputs)
        self.spatial = tf.expand_dims(k, axis=0)

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

        a0, a1 = self.model.predict(self.spatial)
        #probs = tf.nn.softmax(a0)
        #sample = np.random.choice([0, 1, 2, 3], p=probs)

        dist = tf.distributions.Categorical(logits=a0)
        sample = dist.sample()
        if self.debug:
            print("a0 = ", a0, a0.shape)
            print(sample)
        building = int(sample) # now an int between 0 and 3
        if self.debug:
            log("a0 = " + str(a0))

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

    def add_spatial(self, net):
        '''
        Gets the spatial action of the network
        '''
        if self.debug:
            log("getting spatial action")
            s = Stopwatch()
        net = Layers.Conv2D(32, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv")(net)
        net = Layers.Conv2D(1, [1, 1],
            strides=1,
            padding='SAME',
            name="conv1x1")(net)

        logits = Layers.Flatten()(net)
        #probs = Layers.Dense(flat, activation='softmax', name='softmax')(flat)
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
        a0 = Layers.Dense(n_base_actions,
                    name="a0")(non_spatial)

        # TODO: possibly softmax this and then transform it into an int from 0 - 4
        # possibly use tf autoregressive distribution
        ## Do not work with eager

        #dist = self.track_layer(tf.distributions.Categorical(logits=a0))
        #sample = self.track_layer(dist.sample())

        if self.debug:
            log("Finished non-spatial action. Took: " + s.delta)

        return a0

    def add_base(self):
        if self.debug:
            log("Adding base")
            s = Stopwatch()
        with tf.name_scope("adding_base") as scope:
            net = self.input
            for i in range(2):
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
        print(">> SCUD >> Saved model to file ", path)
    
    def load(self, filepath, savename):
        if savename is None:
            path = os.path.join(filepath, str(self.name) + '.h5')
        else:
            if savename.endswith('.h5') == False:
                path = os.path.join(filepath, str(savename) + '.h5')
            else:
                path = os.path.join(filepath, str(savename))
        self.model = tf.keras.models.load_model(path)
        print(">> SCUD >> Model restored from file ", path)


if __name__ == '__main__':

    k = Stopwatch()
    s = Scud('www', debug=True)
    we = s.get_flat_weights()
    log("Round-time was {}".format(k.delta))
    
