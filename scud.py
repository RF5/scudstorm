# -*- coding: utf-8 -*-
'''
Scudstorm

Entelect Challenge 2018
Author: Matthew Baas
'''

import json
import os
import random
import metrics
import tensorflow as tf
from metrics import log, Stopwatch
import numpy as np

'''
Internal agent config
'''
debug = True
debug_verbose = False
endpoints = {}
device = 'cpu'
tf.enable_eager_execution()
# let an example map size be 20x40, so each player's building area is 20x20
    
class Scud(object):
    
    def __init__(self,state_location):
        '''
        Initialize Bot.
        Load all game state information.
        '''
        if debug_verbose and debug:
            log("Testing tensorflow")
            s = Stopwatch()
            print("TensorFlow version: {}".format(tf.VERSION))
            print("Eager execution: {}".format(tf.executing_eagerly()))

            log("Finished, took: " + s.delta)

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

        if debug:
            if debug_verbose:
                log("full_map: " + str(self.full_map))
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
            self.spatial = tf.concat([pb, ob, proj], axis=-1) # 20x20x14
            self.spatial = tf.expand_dims(self.spatial, axis=0)

            if debug:
                log("Done shaping inputs! Took " + s.delta)

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
        log("Running conv2d on " + device)
        with tf.device('/' + device + ':0'):
            net = self.add_base()

        ## loading the state (for RNN stuffs)
        if debug:
            log("Loading state")
            sss = Stopwatch()
        internal_state = np.load('scudstate.npy') # takes ~ 0.031s
        if debug:
            log("State loaded. Took: " + sss.delta)

        ## saving the state (for RNN stuffs)
        if debug:
            log("Saving state")
            ss = Stopwatch()
        new_state = None
        np.save('scudstate.npy')
        if debug:
            log("State saved. Took: " + ss.delta)

        lanes = []
        x,y,building = 0,0,0
        #check all lanes for an attack unit
        for i in range(self.rows):
            if len(self.getUnOccupied(self.player_buildings[i])) == 0:
                #cannot place anything in a lane with no available cells.
                continue
            elif ( self.checkAttack(i) and (self.player_info['energy'] >= self.prices['DEFENSE']) and (self.checkMyDefense(i)) == False):
                #place defense unit if there is an attack building and you can afford a defense building
                lanes.append(i)
        #lanes variable will now contain information about all lanes which have attacking units
        #A count of 0 would mean all lanes are not under attack
        if (len(lanes) > 0) :
            #Chose a random lane under attack to place a defensive unit
            #Chose a cell that is unoccupied in that lane
            building = 0
            y = random.choice(lanes)
            x = random.choice(self.getUnOccupied(self.player_buildings[y]))
        #otherwise, build a random building type at a random unoccupied location
        # if you can afford the most expensive building
        elif  self.player_info['energy'] >= max(s.prices.values()):
            building = random.choice([0,1,2])
            x = random.randint(0,self.rows)
            y = random.randint(0,int(self.columns/2)-1)
        else:
            self.write_no_op()
            return None
        
        self.write_action(x,y,building)
        return x,y,building
    
    def write_action(self,x,y,building):
        '''
        command in form : x,y,building_type
        '''
        outfl = open('command.txt','w')
        outfl.write(','.join([str(x),str(y),str(building)]))
        outfl.close()
        return None

    def write_no_op(self):
        '''
        command in form : x,y,building_type
        '''
        outfl = open('command.txt','w')
        outfl.write("")
        outfl.close()
        return None

    def add_base(self):
        if debug:
            log("Adding base")
            s = Stopwatch()
        with tf.name_scope("adding_base") as scope:
            ss = Stopwatch()
            net = tf.layers.conv2d(self.spatial, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv1") # ok well this takes 5 seconds
            log("1st conv: " + ss.delta)
            ss = Stopwatch()
            net = tf.layers.conv2d(net, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv2") # ok well this takes 5 seconds
            log("2nd conv: " + ss.delta)
            ss = Stopwatch()
            net = tf.layers.conv2d(net, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv3") # ok well this takes 5 seconds
            log("3rd conv: " + ss.delta)
            ss = Stopwatch()
            net = tf.layers.conv2d(net, 32, [3, 3],
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu,
                        name="conv4") # ok well this takes 5 seconds
            log("4th conv: " + ss.delta)


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
    