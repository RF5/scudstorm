'''
Scudstorm metrics

Entelect Challenge 2018
Author: Matthew Baas
'''
import os
import numpy as np
import tensorflow as tf
from metrics import log

debug = True
## Config stuff that relate to the game engine
action_names = ['no_op', 'attack', 'defense', 'energy']

def get_logdir(name=None):
	'''
	returns the log dir corresponding to the supplied name
	'''
	base_dir = os.path.dirname((os.path.abspath(__file__))) # now in scudstorm directory
	if name is not None:
		log_dir = os.path.join(base_dir, 'logs', str(name))
	else:
		log_dir = os.path.join(base_dir, 'logs')

	util_log(log_dir)
	return log_dir

def util_log(msg):
	print(">> UTIL LOG >>\t", msg)

def write_no_op():
	'''
	command in form : x,y,building_type
	'''
	outfl = open('command.txt','w')
	outfl.write("")
	outfl.close()
	return None

def write_action(x,y,building):
	'''
	command in form : x,y,building_type

	if building is no_op (0), then that indicates a NO_OP action and we just write a no op 
	regardless of what x and y are
	'''
	if debug:
		util_log("Writing action: x = " + str(x) + ", y = " + str(y) + "\tBuilding = " + action_names[building])

	if building == 0:
		write_no_op()
		return

	outfl = open('command.txt','w')
	outfl.write(','.join([str(x),str(y),str(building)]))
	outfl.close()
	return None