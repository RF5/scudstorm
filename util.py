'''
Scudstorm metrics

Entelect Challenge 2018
Author: Matthew Baas
'''

import numpy as np
import tensorflow as tf
from metrics import log, Stopwatch

class Profile(object):

	def __init__(self, name):
		self.name = name

	def __enter__(self):
		log("Entering " + str(name))
		s = Stopwatch()

	def __exit__(self):
		log("Exiting " + str(name) + ". Took: " + s.delta)