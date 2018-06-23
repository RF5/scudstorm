'''
Scudstorm process manager -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import sys
import traceback
import multiprocessing
from common.subproc_env_manager import SubprocEnvManager
from pyenv import Env
from common.metrics import Stopwatch
from scud2 import Scud
import storm
import time

# Config vars
n_envs = 5
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
    try:
        env = SubprocEnvManager([make_env(s) for s in env_names])
    except EOFError as e:
        print("caught an EOFError ", e, '\nClosing the env now')
        env.close()
        return
    print('>> manager >> created envs. Took ', s.delta)
    no_act_vec = [(0, 0, 3,) for _ in range(n_envs)]

    # TODO:
    # obs = np.zeros() # some initial state
    if train:
        try:
            storm.train(env, n_envs, no_act_vec)
        except Exception as err:
            try:
                exc_info = sys.exc_info()

            finally:
                traceback.print_exception(*exc_info)
                del exc_info
        finally:
            print('>> manager >> closing env. Total runtime: ', sTime.delta)
            env.close()
            sys.exit(0)
    else:
        actions = no_act_vec
        agents = [Scud(name=str(i), debug=False) for i in range(n_envs)]
        refbot = Scud('ref', False)
        env.reset()
        ob = env.get_base_obs()
        ref_act = None
        for i in range(1):
            ss = Stopwatch()
            print(">> manager >> step {}, taking actions: {}".format(i, actions))
            obs, rews = env.step(actions, ref_act) # obs is n_envs x 1
            #print(rews)
            #print("obs shape", obs.shape)
            try:
                actions = [agent.step(obs[i][0]) for i, agent in enumerate(agents)]
                ref_act = [refbot.step(obs[i][1]) for i in range(len(agents))]
            except TypeError as e:
                print("TypeError!!! ", e)
                break
            print('>> manager >> just took step {}. Took: {}'.format(i, ss.delta))
            time.sleep(0.03)
    # gets all the variables of the model
    # all_variables = agents[0].model.get_weights()
    
    print('>> manager >> closing env. Total runtime: ', sTime.delta)
    env.close()
    sys.exit(0)

if __name__ == '__main__':
    main()