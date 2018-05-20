'''
Scudstorm training orchestrator -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''
import tensorflow as tf
import common.util
import os
import argparse
import time
from common.metrics import Stopwatch
from scud2 import Scud
import numpy as np

summary = tf.contrib.summary

##############################
###### TRAINING CONFIG #######
n_steps = 50
n_generations = 2
n_elite = 2

sigma = 0.002
##############################

def train(env, n_envs, no_op_vec):
    print(str('='*50) + '\n' + 'Initializing agents\n' + str('='*50) )
    
    actions = no_op_vec
    ## TODO: change agent layers to use xavier initializer
    agents = [Scud(name=str(i), debug=False) for i in range(n_envs)]
    n_params = agents[0].model.count_params()

    print(str('='*50) + '\n' + 'Beginning training\n' + str('='*50) )
    scores = []
    for g in range(n_generations):
        print(str('='*50) + '\n' + 'Generation ' + str(g) + '\n' + str('='*50) )
        for i in range(n_envs):

            parent_ind = np.random.randint(n_elite)
            offspring = mutate(agents[parent_ind], g)

            scores.append(rollout(env, offspring))

        ss = Stopwatch()
        print(">> storm >> taking actions: ", actions)
        obs = env.step(actions) # obs is n_envs x 1
        
        actions = [agent.step(obs[i][0]) for i, agent in enumerate(agents)]
        print('>> storm >> just took step {}. Took: {}'.format(i, ss.delta))

def mutate(agent, g):
    old_params = agent.get_flat_weights()
    new_params = []
    for param in old_params:
        new_params.append(param + sigma*np.random.randn(*param.shape))
    new_agent = Scud(agent.name + "gen" + str(g))
    new_agent.set_flat_weights(new_params)
    return agent

def rollout(env, agent):
    return 1


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
