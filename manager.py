'''
Scudstorm process manager -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import multiprocessing
from common.subproc_env_manager import SubprocEnvManager
from pyenv import Env

# Config vars
n_envs = 4
console_debug = True

def main():
    env_names = ['env' + str(i) for i in range(n_envs)]

    def make_env(name):
        def env_fn():
            env_inside = Env(name, console_debug)

            return env_inside

        return env_fn
    print('>> manager >> creating envs')
    env = SubprocEnvManager([make_env(s) for s in env_names])

    no_act_vec = [(0, 0, 3,) for _ in range(n_envs)]

    for i in range(4):
        print('>> manager >> taking step ', i)
        env.step(no_act_vec)

    env.close()

if __name__ == '__main__':
    main()