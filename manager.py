'''
Scudstorm process manager -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import multiprocessing
from common.subproc_env_manager import SubprocEnvManager
from pyenv import Env
from common.metrics import Stopwatch
from scud2 import Scud
import storm
import time

# Config vars
n_envs = 3
console_debug = False
train = True

def main():
    sTime = Stopwatch()
    env_names = ['env' + str(i) for i in range(n_envs)]

    def make_env(name):
        def env_fn():
            env_inside = Env(name, console_debug)

            return env_inside

        return env_fn
    print('>> manager >> creating envs')
    s = Stopwatch()
    env = SubprocEnvManager([make_env(s) for s in env_names])
    print('>> manager >> created envs. Took ', s.delta)
    no_act_vec = [(0, 0, 3,) for _ in range(n_envs)]

    # TODO:
    # obs = np.zeros() # some initial state
    if train:
        storm.train(env, n_envs, no_act_vec)
    else:
        actions = no_act_vec
        agents = [Scud(name=str(i), debug=False) for i in range(n_envs)]
        for i in range(4):
            ss = Stopwatch()
            print(">> manager >> taking actions: ", actions)
            obs = env.step(actions) # obs is n_envs x 1
            try:
                actions = [agent.step(obs[i][0]) for i, agent in enumerate(agents)]
            except TypeError as e:
                print("Got ", e)
                break
            print('>> manager >> just took step {}. Took: {}'.format(i, ss.delta))
    # gets all the variables of the model
    # all_variables = agents[0].model.get_weights()

    print('>> manager >> closing env. Total runtime: ', sTime.delta)
    env.close()

if __name__ == '__main__':
    main()