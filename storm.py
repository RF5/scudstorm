'''
Scudstorm training orchestrator -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''
import tensorflow as tf
import util
import os
import argparse
import time
from metrics import Stopwatch
from scud import Scud

summary = tf.contrib.summary

##############################
###### TRAINING CONFIG #######
n_steps = 50

##############################

class Storm(object):

    def __init__(self, train=False):
        # setting up logs
        self.logdir = util.get_logdir('test')
        self.writer = summary.create_file_writer(util.get_logdir('test'), flush_millis=10000)
        self.writer.set_as_default()
        self.global_step = tf.train.get_or_create_global_step()
        # Does not really work. TODO: find a way to get graph in eager mode
        #tf.contrib.summary.graph(tf.get_default_graph())
        self.is_training = train
        self.persist()
        self.act_on_arrival()

    def persist(self):
        while True:
            with open('running_state.txt', 'r') as f:
                run_state = int(f.read())
                print("read in: ", run_state)
            if run_state == 2:
                break
            if run_state == 1:
                self.act_on_arrival()
            time.sleep(0.1)

    def act_on_arrival(self):
        assert self.is_training == False, "We should not be training if ur doing this"
        with open('running_state.txt', 'w') as f:
            f.write('2')
        self.infer()
        with open('running_state.txt', 'w') as f:
            f.write('0') 
        return


    def infer(self):
        print("Beginning storm...")
        stopwatch = Stopwatch()

        self.global_step.assign_add(1)
        s = Scud('state.json')
        x,y, building = s.generate_action()

        # taking logs
        with summary.always_record_summaries(): 
        #with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            summary.histogram('outputs/buildings', building)
            summary.histogram('outputs/x', x)
            summary.histogram('outputs/y', y)
        
        summary.flush()
        print("Total time for inference was {}".format(stopwatch.delta))

    @staticmethod
    def main():
        print("Beginning storm...")
        stopwatch = Stopwatch()

        global_step = tf.train.get_or_create_global_step()
        for i in range(n_steps):
            global_step.assign_add(1)
            s = Scud('state.json')
            x,y, building = s.generate_action()

            # taking logs
            with summary.always_record_summaries(): 
            #with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                summary.histogram('outputs/buildings', building)
                summary.histogram('outputs/x', x)
                summary.histogram('outputs/y', y)
        
        summary.flush()
        print("Total time for {} steps was {}".format(n_steps, stopwatch.delta))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-vis", type=bool, default=False, help="should we visualize the results in tensorboard upon completion")
    args = parser.parse_args()
    if args.vis == True:
        kk = Storm()
    else:
        Storm.main()
