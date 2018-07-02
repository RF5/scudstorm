'''
Scudstorm observation parsing

Entelect Challenge 2018
Author: Matthew Baas
'''
import numpy as np
import tensorflow as tf
from common.metrics import Stopwatch

debug = False

def parse_obs(game_state):
    full_map = game_state['gameMap']
    rows = game_state['gameDetails']['mapHeight']
    columns = game_state['gameDetails']['mapWidth']
    
    player_buildings = getPlayerBuildings(full_map, rows, columns)
    opponent_buildings = getOpponentBuildings(full_map, rows, columns)
    projectiles = getProjectiles(full_map, rows, columns)
    
    player_info = getPlayerInfo('A', game_state)
    opponent_info = getPlayerInfo('B', game_state)
    
    round_num = game_state['gameDetails']['round']

    # works for jar v1.1.2
    prices = {"ATTACK": game_state['gameDetails']['buildingsStats']['ATTACK']['price'],
                "DEFENSE":game_state['gameDetails']['buildingsStats']['DEFENSE']['price'],
                "ENERGY":game_state['gameDetails']['buildingsStats']['ENERGY']['price']}

    with tf.name_scope("shaping_inputs") as scope:
        if debug:
            print("Shaping inputs...")
            s = Stopwatch()

        pb = tf.one_hot(indices=player_buildings, depth=5, axis=-1, name="player_buildings") # 20x20x5
        ob = tf.one_hot(indices=opponent_buildings, depth=5, axis=-1, name="opp_buildings") # 20x20x5
        proj = tf.one_hot(indices=projectiles, depth=3, axis=-1, name='projectiles') # 20x40x3
        k = proj.get_shape().as_list()
        proj = tf.reshape(proj, [int(k[0]), int(k[1] / 2), 6]) # 20x20x6. Only works for single misssiles

        non_spatial = list(player_info.values())[1:] + list(opponent_info.values())[1:] + list(prices.values()) # 11x1
        non_spatial = tf.cast(non_spatial, dtype=tf.float32)
        # broadcasting the non-spatial features to the channel dimension
        broadcast_stats = tf.tile(tf.expand_dims(tf.expand_dims(non_spatial, axis=0), axis=0), [int(k[0]), int(k[1] / 2), 1]) # now 20x20x11

        # adding all the inputs together via the channel dimension
        spatial = tf.concat([pb, ob, proj, broadcast_stats], axis=-1) # 20x20x(16 + 11)
        

        if debug:
            print("Finished shaping inputs. Took " + s.delta + "\nShape of inputs:" +  str(spatial.shape))

        return spatial, rows, columns

def getPlayerInfo(playerType, game_state):
    '''
    Gets the player information of specified player type
    '''
    for i in range(len(game_state['players'])):
        if game_state['players'][i]['playerType'] == playerType:
            return game_state['players'][i]
        else:
            continue        
    return None

def getOpponentBuildings(full_map, rows, columns):
    '''
    Looks for all buildings, regardless if completed or not.
    0 - Nothing
    1 - Attack Unit
    2 - Defense Unit
    3 - Energy Unit
    '''
    opponent_buildings = []
    
    for row in range(0,rows):
        buildings = []
        for col in range(int(columns/2),columns):
            if (len(full_map[row][col]['buildings']) == 0):
                buildings.append(0)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'ATTACK'):
                buildings.append(1)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'DEFENSE'):
                buildings.append(2)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'ENERGY'):
                buildings.append(3)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'TESLA'):
                buildings.append(4)
            else:
                buildings.append(0)
            
        opponent_buildings.append(buildings)
        
    return opponent_buildings

def getPlayerBuildings(full_map, rows, columns):
    '''
    Looks for all buildings, regardless if completed or not.
    0 - Nothing
    1 - Attack Unit
    2 - Defense Unit
    3 - Energy Unit
    '''
    player_buildings = []
    
    for row in range(0,rows):
        buildings = []
        for col in range(0,int(columns/2)):
            if (len(full_map[row][col]['buildings']) == 0):
                buildings.append(0)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'ATTACK'):
                buildings.append(1)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'DEFENSE'):
                buildings.append(2)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'ENERGY'):
                buildings.append(3)
            elif (full_map[row][col]['buildings'][0]['buildingType'] == 'TESLA'):
                buildings.append(4)
            else:
                buildings.append(0)
            
        player_buildings.append(buildings)
        
    return player_buildings

def getProjectiles(full_map, rows, columns):
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

    for row in range(0,rows):
        temp = []
        for col in range(0,columns):
            if (len(full_map[row][col]['missiles']) == 0):
                temp.append(0)
            elif (full_map[row][col]['missiles'][0]['playerType'] == 'A'):
                temp.append(1)
            elif (full_map[row][col]['missiles'][0]['playerType'] == 'B'):
                temp.append(2)
            
        projectiles.append(temp)
        
    return projectiles