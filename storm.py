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
n_generations = 5
trunc_size = 2

max_steps_per_eval = 10
gamma = 0.99 # reward decay

sigma = 0.002
##############################

def train(env, n_envs, no_op_vec):
    print(str('='*50) + '\n' + 'Initializing agents\n' + str('='*50) )
    
    #actions = no_op_vec
    ## TODO: change agent layers to use xavier initializer
    agents = [Scud(name=str(i), debug=False) for i in range(n_envs)]
    #n_params = agents[0].model.count_params()
    elite = agents[0]

    print(str('='*50) + '\n' + 'Beginning training\n' + str('='*50) )

    for g in range(n_generations):
        print(str('='*50) + '\n' + 'Generation ' + str(g) + '\n' + str('='*50) )

        agents = collective_rollout(env, agents)

        agents = sorted(agents, key = lambda agent : agent.fitness_score, reverse=True)
        elite = agents[0]
        for a in agents:
            print(a.fitness_score)

        #del agents[trunc_size:]
        ind = trunc_size

        while ind < n_envs:
            parent_ind = np.random.randint(trunc_size)
            mutate(agents[parent_ind], agents[ind], g)

            ind += 1
        # for i in range(n_envs):
        #     print("mutating agent ", i)
        #     parent_ind = np.random.randint(n_elite)
        #     offspring = mutate(agents[parent_ind], g)

        #obs = env.step(actions) # obs is n_envs x 1
       

def mutate(parent, child, g):
    old_params = parent.get_flat_weights()
    new_params = []
    print(">> storm >> mutating agent: ", parent.name)
    for param in old_params:
        new_params.append(param + sigma*np.random.randn(*param.shape))
    child.name = parent.name + "gen" + str(g)
    child.set_flat_weights(new_params)
    child.fitness_score = 0

def collective_rollout(env, agents):
    step = 0
    actions = [(0, 0, 3,) for _ in range(len(agents))]
    suc = env.reset()
    if all(suc) == False:
        print("something fucked out. Could not reset all envs.")
        return
    obs = env.get_base_obs()
    for a in agents:
        a.fitness_score = 0

    while step < max_steps_per_eval:
        ss = Stopwatch()
        
        actions = [agent.step(obs[i]) for i, agent in enumerate(agents)]
        print(">> storm >> taking actions: ", actions)
        obs, rews = env.step(actions)
        
        for i, a in enumerate(agents):
            a.fitness_score = rews[i] + gamma*a.fitness_score

        print('>> storm >> just took step {}. Took: {}'.format(step, ss.delta))
        step = step + 1

    return agents


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
