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
n_envs = 1
console_debug = False

def main():
    env_names = ['env' + str(i) for i in range(n_envs)]

    def make_env(name):
        def env_fn():
            env_inside = Env(name, console_debug)

            return env_inside

        return env_fn
    print('MANAGER: creating envs')
    env = SubprocEnvManager([make_env(s) for s in env_names])
    test_actions = [(0, 0, 0), (0, 0, 3), (1, 2, 2)]
    print('manager: taking step')
    env.step(test_actions)
    print('manager: attempting to terminate')
    env.close()


if __name__ == '__main__':
    main()