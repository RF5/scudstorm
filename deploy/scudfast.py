'''
Scudstorm fast - for tournamet runtime
Upgrades:
- Camo Netting
- Anthrax Beta

Currently takes ~1.51s when running on medium-low spec PC. (GTX 680, i7-3770k)
Any reasonably decent server runtime environment should give performance
required to run within 2s

Author: Matthew Baas
'''

import json
import os
import tensorflow as tf
from custom_layers import SampleCategoricalLayer, OneHotLayer

tf.enable_eager_execution()

##########################
## Game config stuff
map_width = 16
rows = map_width

map_height = 8
columns = map_height

action_map = {'defence': 0, 'attack': 1, 'energy': 2, 'deconstruct': 3, 'tesla': 4, 'no_op': 5}

reverse_action_map = {
    action_map['defence'] : 'defence', 
    action_map['attack']: 'attack', 
    action_map['energy']: 'energy',
    action_map['deconstruct']: 'deconstruct',
    action_map['tesla']: 'tesla',
    action_map['no_op']: 'no_op'}

custom_keras_layers = {
    'SampleCategoricalLayer': SampleCategoricalLayer,
    'OneHotLayer': OneHotLayer}

debug = True

class ScudFast:
    
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
        
        self.prices = {"ATTACK": self.game_state['gameDetails']['buildingsStats']['ATTACK']['price'],
                    "DEFENSE":self.game_state['gameDetails']['buildingsStats']['DEFENSE']['price'],
                    "ENERGY":self.game_state['gameDetails']['buildingsStats']['ENERGY']['price'],
                    "TESLA":self.game_state['gameDetails']['buildingsStats']['TESLA']['price']}

        path = self.get_savepath()
        self.model = tf.keras.models.load_model(path, custom_objects=custom_keras_layers)
        self.form_input()

        return None
    
    def form_input(self):
        with tf.name_scope("shaping_inputs") as scope:

            pb = tf.one_hot(indices=self.player_buildings, depth=5, axis=-1, name="player_buildings") # 20x20x5
            ob = tf.one_hot(indices=self.opponent_buildings, depth=5, axis=-1, name="opp_buildings") # 20x20x5
            proj = tf.one_hot(indices=self.projectiles, depth=3, axis=-1, name='projectiles') # 20x40x3
            k = proj.get_shape().as_list()
            proj = tf.reshape(proj, [int(k[0]), int(k[1] / 2), 6]) # 20x20x6. Only works for single misssiles

            self.non_spatial = list(self.player_info.values())[1:] + list(self.opponent_info.values())[1:] + list(self.prices.values()) # 11x1
            self.non_spatial = tf.cast(self.non_spatial, dtype=tf.float32)
            # broadcasting the non-spatial features to the channel dimension
            broadcast_stats = tf.tile(tf.expand_dims(tf.expand_dims(self.non_spatial, axis=0), axis=0), [int(k[0]), int(k[1] / 2), 1]) # now 20x20x11

            # adding all the inputs together via the channel dimension
            self.spatial = tf.concat([pb, ob, proj, broadcast_stats], axis=-1) # 20x20x(16 + 11)
            self.spatial = tf.expand_dims(self.spatial, axis=0)

    def get_savepath(self):
        return 'scudsave.h5'
        
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
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'TESLA'):
                    buildings.append(4)
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
                elif (self.full_map[row][col]['buildings'][0]['buildingType'] == 'TESLA'):
                    buildings.append(4)
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
        
    def generateAction(self):
        # spatial now should have shape (1, 8, 8, 25)
        a0, a1 = self.model.predict(self.spatial)
        building = a0[0]

        coords = tf.unravel_index(a1[0], [self.rows, self.columns/2])
        x = int(coords[0])
        y = int(coords[1])
        
        self.writeCommand(x,y,building)

        return x,y,building
    
    def writeCommand(self,x,y,building):
        '''
        command in form : x,y,building_type
        '''
        if debug:
            print("Action: X: {} Y: {} BUILDING: {}".format(x, y, reverse_action_map[building]))

        outfl = open('command.txt','w')

        if reverse_action_map[building] == 'no_op':
            outfl.write("")
        else:
            outfl.write(','.join([str(x),str(y),str(building)]))

        outfl.close()
        return None

if __name__ == '__main__':
    if debug:
        import time
        startime = int(round(time.time() * 1000))
    s = ScudFast('state.json')
    s.generateAction()
    if debug:
        endtime = int(round(time.time() * 1000))
        print("Scudfast took: ", str((endtime - startime) / 1000) + 's')
    