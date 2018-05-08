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

'''
Internal agent config
'''
debug = True
n_base_actions = 4 # number of base actions -- 0=NO OP, 1=DEFENSE, 2=OFFENSE, 3=ENERGY...
debug_verbose = False
endpoints = {}
device = 'cpu'
tf.enable_eager_execution()
# let an example map size be 20x40, so each player's building area is 20x20

if debug_verbose and debug:
    log("Testing tensorflow")
    s = Stopwatch()
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    log("Finished, took: " + s.delta)
    
class Scud(object):
    
    def __init__(self,state_location):
        '''
        Initialize Bot.
        Load all game state information.
        '''

        try:
            self.game_state = self.loadState(state_location)
        except IOError:
            print("Cannot load Game State")
            
        self.full_map = self.game_state['gameMap']
        self.rows = self.game_state['gameDetails']['mapHeight']
        self.columns = self.game_state['gameDetails']['mapWidth']
        self.command = ''
        
        self.player_buildings = self.getPlayerBuildings()
        self.opponent_buildings = self.getOpponentBuildings()
        self.projectiles = self.getProjectiles()
        
        self.player_info = self.getPlayerInfo('A')
        self.opponent_info = self.getPlayerInfo('B')
        
        self.round = self.game_state['gameDetails']['round']
        
        self.prices = {"ATTACK":self.game_state['gameDetails']['buildingPrices']['ATTACK'],
                       "DEFENSE":self.game_state['gameDetails']['buildingPrices']['DEFENSE'],
                       "ENERGY":self.game_state['gameDetails']['buildingPrices']['ENERGY']}

        if debug and debug_verbose:
            log("rows: " + str(self.rows))
            log("columns: " + str(self.columns))
            log("player_buildings: " + str(self.player_buildings))
            log("opp_buildings: " + str(self.opponent_buildings))
            log("projectiles: " + str(self.projectiles))
            log("player_info: " + str(self.player_info))
            log("opp_info: " + str(self.opponent_info))
            log("Round: " + str(self.round))
            log("Prices: " + str(self.prices))

        # getting inputs
        with tf.name_scope("shaping_inputs") as scope:
            if debug:
                log("Shaping inputs...")
                s = Stopwatch()

            pb = tf.one_hot(indices=self.player_buildings, depth=4, axis=-1, name="player_buildings") # 20x20x4
            ob = tf.one_hot(indices=self.opponent_buildings, depth=4, axis=-1, name="opp_buildings") # 20x20x4
            proj = tf.one_hot(indices=self.projectiles, depth=3, axis=-1, name='projectiles') # 20x40x3
            k = proj.get_shape().as_list()
            proj = tf.reshape(proj, [k[0], k[1] / 2, 6]) # 20x20x6. Only works for single misssiles

            self.non_spatial = list(self.player_info.values())[1:] + list(self.opponent_info.values())[1:] + list(self.prices.values()) # 11x1
            self.non_spatial = tf.cast(self.non_spatial, dtype=tf.float32)
            # broadcasting the non-spatial features to the channel dimension
            broadcast_stats = tf.tile(tf.expand_dims(tf.expand_dims(self.non_spatial, axis=0), axis=0), [k[0], k[1] / 2, 1]) # now 20x20x11

            # adding all the inputs together via the channel dimension
            self.spatial = tf.concat([pb, ob, proj, broadcast_stats], axis=-1) # 20x20x(14 + 11)
            self.spatial = tf.expand_dims(self.spatial, axis=0)

            if debug:
                log("Finished shaping inputs. Took " + s.delta)

        return None

    def loadState(self,state_location):
        '''
        Gets the current Game State json file.
        '''
        return json.load(open(state_location,'r'))

    def getPlayerInfo(self,playerType):
        '''
        Gets the player information of specified player type
        '''
        for i in range(len(self.game_state['players'])):
            if self.game_state['players'][i]['playerType'] == playerType:
                return self.game_state['players'][i]
            else:
                continue        
        return None
    
    def getOpponentBuildings(self):
        '''
        Looks for all buildings, regardless if completed or not.
        0 - Nothing
        1 - Attack Unit
        2 - Defense Unit
        3 - Energy Unit
        '''
        opponent_buildings = []
        
        for row in range(0,self.rows):
            buildings = []
            for col in range(int(self.columns/2),self.columns):
                if (len(self.full_map[row][col]['buildings']) == 0):
                    buildings.append(0)
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'ATTACK'):
                    buildings.append(1)
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'DEFENSE'):
                    buildings.append(2)
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'ENERGY'):
                    buildings.append(3)
                else:
                    buildings.append(0)
                
            opponent_buildings.append(buildings)
            
        return opponent_buildings
    
    def getPlayerBuildings(self):
        '''
        Looks for all buildings, regardless if completed or not.
        0 - Nothing
        1 - Attack Unit
        2 - Defense Unit
        3 - Energy Unit
        '''
        player_buildings = []
        
        for row in range(0,self.rows):
            buildings = []
            for col in range(0,int(self.columns/2)):
                if (len(self.full_map[row][col]['buildings']) == 0):
                    buildings.append(0)
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'ATTACK'):
                    buildings.append(1)
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'DEFENSE'):
                    buildings.append(2)
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'ENERGY'):
                    buildings.append(3)
                else:
                    buildings.append(0)
                
            player_buildings.append(buildings)
            
        return player_buildings
    
    def getProjectiles(self):
        '''
        Find all projectiles on the map.
        0 - Nothing there
        1 - Projectile belongs to player
        2 - Projectile belongs to opponent
        '''
        projectiles = []
        
        ## TODO: make this somehow capture multiple missiles
        # that is controlled in the ...['missiles'][0] part, where 0
        # could be many missiles. Possibly make multiple missiles be double the one-hot value?
        # or just stack for channels for 2-missiles and 3-missiles?

        for row in range(0,self.rows):
            temp = []
            for col in range(0,self.columns):
                if (len(self.full_map[row][col]['missiles']) == 0):
                    temp.append(0)
                elif (self.full_map[row][col]['missiles'][0]['playerType'] == 'A'):
                    temp.append(1)
                elif (self.full_map[row][col]['missiles'][0]['playerType'] == 'B'):
                    temp.append(2)
                
            projectiles.append(temp)
            
        return projectiles
    

    def checkDefense(self, lane_number):
        '''
        Checks a lane.
        Returns True if lane contains defense unit.
        '''
        
        lane = list(self.opponent_buildings[lane_number])
        if (lane.count(2) > 0):
            return True
        else:
            return False

    def checkMyDefense(self, lane_number):

        '''
        Checks a lane.
        Returns True if lane contains defense unit.
        '''
        
        lane = list(self.player_buildings[lane_number])
        if (lane.count(2) > 0):
            return True
        else:
            return False
    
    def checkAttack(self, lane_number):

        '''
        Checks a lane.
        Returns True if lane contains attack unit.
        '''
        
        lane = list(self.opponent_buildings[lane_number])
        if (lane.count(1) > 0):
            return True
        else:
            return False
    
    def getUnOccupied(self,lane):
        '''
        Returns index of all unoccupied cells in a lane
        '''
        indexes = []
        for i in range(len(lane)):
            if lane[i] == 0 :
                indexes.append(i)
        
        return indexes
                
        
    def generate_action(self):
        '''
        Scud model estimator
        '''
        if debug:
            log("Running conv2d on " + device)
        with tf.device('/' + device + ':0'):
            net = self.add_base()
            #print("state shape: ", net.shape) # (1, 20, 20, 32)

        ## split into non-spatial and spatial action path
        a0 = self.get_non_spatial(net)
        building = int(a0) # now an int between 0 and 3
        if debug:
            log("a0 = " + str(a0))

        coords = self.get_spatial(net)
        x = int(coords[0])
        y = int(coords[1])
        if debug:
            log("x, y = " + str(x) + ", " + str(y))

        ## loading the state (for RNN stuffs)
        if debug:
            log("Loading state")
            sss = Stopwatch()
        _ = np.load('scudstate.npy') # takes ~ 0.031s
        if debug:
            log("State loaded. Took: " + sss.delta)

        ## saving the state (for RNN stuffs)
        if debug:
            log("Saving state")
            ss = Stopwatch()
        new_state = net
        np.save('scudstate.npy', new_state)
        if debug:
            log("State saved. Took: " + ss.delta)
        
        util.write_action(x,y,building)
        return x,y,building


    def get_spatial(self, net):
        '''
        Gets the spatial action of the network
        '''
        if debug:
            log("getting spatial action")
            s = Stopwatch()
        net = tf.layers.conv2d(self.spatial, 32, [3, 3],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            name="finalConv")
        net = tf.layers.conv2d(net, 1, [1, 1],
            strides=1,
            padding='SAME',
            name="conv1x1")

        flat = tf.layers.flatten(net)
        dist = tf.distributions.Categorical(logits=flat)
        sample = dist.sample()

        coords = tf.unravel_index(sample, [self.rows, self.columns/2])

        if debug:
            log("Finished spatial action inference. Took: " + s.delta)
        return coords


    def get_non_spatial(self, net):
        '''
        Infers the non-spatial action of the network
        '''
        if debug:
            log("Getting non-spatial action")
            s = Stopwatch()
        non_spatial = tf.layers.dense(tf.layers.flatten(net), 256,
                    activation=tf.nn.relu,
                    name="non_spatial")
        a0 = tf.layers.dense(non_spatial, n_base_actions,
                    name="a0")

        # TODO: possibly softmax this and then transform it into an int from 0 - 4
        # possibly use tf autoregressive distribution
        dist = tf.distributions.Categorical(logits=a0)
        sample = dist.sample()

        if debug:
            log("Finished non-spatial action. Took: " + s.delta)

        return sample
        

    def add_base(self):
        if debug:
            log("Adding base")
            s = Stopwatch()
        with tf.name_scope("adding_base") as scope:
            net = tf.layers.conv2d(self.spatial, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv1") # ok well this takes 5 seconds
            net = tf.layers.conv2d(net, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv2") # ok well this takes 5 seconds
            net = tf.layers.conv2d(net, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv3") # ok well this takes 5 seconds
            net = tf.layers.conv2d(net, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv4") # ok well this takes 5 seconds

        if debug:
            log("Finished adding base. Took: " + s.delta)

        return net

if __name__ == '__main__':
    if debug:
        k = Stopwatch()
    s = Scud('state.json')
    s.generate_action()
    if debug:
        log("Round-time was {}".format(k.delta))
    
