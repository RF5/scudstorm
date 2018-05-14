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

# Config vars
n_envs = 1
console_debug = False

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
    actions = no_act_vec
    for i in range(1):
        ss = Stopwatch()
        print("taking actions: ", actions)
        obs = env.step(actions) # obs is n_envs x 1
        agents = [Scud(name=str(i), debug=True) for i in range(n_envs)]
        
        actions = [agent(obs[i][0]) for i, agent in enumerate(agents)]
        agents[0].train_vars()
        print('>> manager >> just took step {}. Took: {}'.format(i, ss.delta))

    print('>> manager >> closing env. Took: ', sTime.delta)
    env.close()

if __name__ == '__main__':
    main()