'''
Scudstorm metrics

Entelect Challenge 2018
Author: Matthew Baas
'''
import os
import numpy as np
import sys
from common.metrics import log
import json

#debug = True
## Config stuff that relate to the game engine
action_names = ['attack', 'defense', 'energy', 'no_op']

def get_logdir(name=None):
	'''
	returns the log dir corresponding to the supplied name
	'''
	base_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) # now in scudstorm directory
	if name is not None:
		log_dir = os.path.join(base_dir, 'logs', str(name))
	else:
		log_dir = os.path.join(base_dir, 'logs')

	util_log(log_dir)
	return log_dir

def get_savedir(name=None):
	'''
	returns the save dir corresponding to the supplied name
	'''
	base_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) # now in scudstorm directory
	if name is not None:
		save_dir = os.path.join(base_dir, 'saves', str(name))
	else:
		save_dir = os.path.join(base_dir, 'saves')

	util_log("saving to " + str(save_dir))
	return save_dir

def util_log(msg):
	print(">> UTIL LOG >>\t", msg)

def write_prep_action(x,y,building, path, debug=True):
	if debug:
		util_log("Writing action: x = " + str(x) + ", y = " + str(y) + "\tBuilding = " + action_names[building] + "\tTo:")
		print(os.path.join(path, 'command2.txt'))

	outfl = open(os.path.join(path, 'command2.txt'),'w')

	if action_names[building] == 'no_op':
		outfl.write("NO_OP")
	else:
		outfl.write(','.join([str(x),str(y),str(building)]))
	outfl.close()
	return

def write_action(x,y,building, path, debug=True):
	'''
	command in form : x,y,building_type

	if building is no_op (0), then that indicates a NO_OP action and we just write a no op 
	regardless of what x and y are
	'''
	if debug:
		util_log("Writing action: x = " + str(x) + ", y = " + str(y) + "\tBuilding = " + action_names[building] + "\tTo:")
		print(os.path.join(path, 'command.txt'))

	outfl = open('command.txt','w')
	if action_names[building] == 'no_op':
		outfl.write("")
	else:	
		outfl.write(','.join([str(x),str(y),str(building)]))
	
	outfl.close()
	return

def get_initial_obs(n_envs):
	lys = []
	for _ in range(n_envs):
		zero_state_file = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'initial_state.json') # now in common dir
		k = json.load(open(zero_state_file,'r'))
		n = json.load(open(zero_state_file,'r'))
		x = np.asarray([k,])
		y = np.asarray([n,])
		intermediate = np.concatenate([x, y], axis=-1)
		lys.append(intermediate)
	return np.stack(lys, axis=0)

class ControlObject(object):
	def __init__(self, code):
		self.code = code