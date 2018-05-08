'''
Scudstorm metrics

Entelect Challenge 2018
Author: Matthew Baas
'''
import time

def log(msg):
	print(">> SCUDSTORM LOG >>\t", msg)

class Stopwatch(object):

	def __init__(self):
		# start time in miliseconds
		self.startime = int(round(time.time() * 1000))

	@property
	def delta(self):
		endtime = int(round(time.time() * 1000))
		return str((endtime - self.startime) / 1000) + 's'