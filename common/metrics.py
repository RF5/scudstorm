'''
Scudstorm metrics

Entelect Challenge 2018
Author: Matthew Baas
'''
import time
import sys
import numpy as np

def log(msg):
	print(">> SCUDSTORM LOG >>\t", msg)

class Stopwatch(object):

	def __init__(self):
		# start time in miliseconds
		self.startime = int(round(time.time() * 1000))
		self.waypoints = []
		self.prev_lap_total_time = 0

	@property
	def delta(self):
		endtime = int(round(time.time() * 1000))
		return str((endtime - self.startime) / 1000) + 's'

	def deltaT(self):
		endT = int(round(time.time() * 1000))
		return (endT - self.startime)/1000
	
	def reset(self):
		self.startime = int(round(time.time() * 1000))

	def lap(self, tag):
		t = self.deltaT()
		self.waypoints.append((t - self.prev_lap_total_time, tag))
		self.prev_lap_total_time += t - self.prev_lap_total_time

	def print_results(self):
		total_time = self.deltaT()
		for t, tag in self.waypoints:
			print(('--> {:30} : {:.2f}% ({:4.2f}s)').format(tag, 100*(t/total_time) , t))
		print('--> Total time: ', total_time)
		
class ProgressBar(object):

	def __init__(self, total):
		self.total = total
		self.cur_i = 0

	def show(self, i):
		frac = i/self.total
		length = int(frac * 21)
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*length, 100*frac))
		sys.stdout.flush()
	
	def increment(self, inc_num):
		self.cur_i += inc_num
		self.show(self.cur_i)

	def close(self):
		frac = 1
		length = int(frac * 20)
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*length, 100*frac))
		sys.stdout.flush()
		print()

class MovingAverage(object):
	def __init__(self, n):
		self.arr = [0 for _ in range(n)]

	def push(self, val):
		del self.arr[0]
		self.arr.append(val)

	def mean(self):
		return np.mean(self.arr)

	def value(self):
		return self.mean()