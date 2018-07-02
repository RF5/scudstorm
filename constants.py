'''
Scudstorm constants

Config stuff that relate to the game engine

Entelect Challenge 2018
Author: Matthew Baas
'''

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

no_op_action = (0, 0, action_map['no_op'],)

n_base_actions = 6