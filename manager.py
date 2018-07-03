'''
Scudstorm process manager -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import sys
import traceback
import argparse
from common.subproc_env_manager import SubprocEnvManager
from pyenv import Env
from common.metrics import Stopwatch
from scud2 import Scud
import storm
import time
import runner
from common import util
import constants

# Config vars
n_envs = 1
console_debug = False
train = True
mode_options = ['train', 'resume', 'test']

def main(mode):
    sTime = Stopwatch()
    env_names = ['env' + str(i) for i in range(n_envs)]

    if mode == 'test':
        train = False
    else:
        train = True
    if mode == 'resume':
        resume_training = True
    else: 
        resume_training = False

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
    no_act_vec = [constants.no_op_action for _ in range(n_envs)]

    # TODO:
    # obs = np.zeros() # some initial state
    if train:
        try:
            storm.train(env, n_envs, no_act_vec, resume_training)
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
        try:
            actions = no_act_vec
            agents = [Scud(name=str(i), debug=False) for i in range(n_envs)]
            refbot = Scud('ref', False)
            env.reset()
            obs = util.get_initial_obs(n_envs)
            #print("manager obs shape = ", ob.shape)
            ref_act = None
            for i in range(5):
                ss = Stopwatch()
                
                #print(rews)
                #print("obs shape", obs.shape)
                try:
                    sss = Stopwatch()
                    actions = [agent.step(obs[i][0]) for i, agent in enumerate(agents)]
                    ref_act = [refbot.step(obs[i][1]) for i in range(len(agents))]
                except TypeError as e:
                    print("TypeError!!! ", e)
                    break

                print("running NN :", sss.delta)
                print(">> manager >> step {}, taking actions: {}".format(i, actions))
                ssss = Stopwatch()
                obs, rews, infos = env.step(actions, ref_act) # obs is n_envs x 1
                print("Running env : ", ssss.delta)
                print('>> manager >> just took step {}. Took: {}'.format(i, ss.delta))
                time.sleep(0.03)

            #runner.run_battle(agents[0], refbot, env)

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
    # gets all the variables of the model
    # all_variables = agents[0].model.get_weights()
    
    print('>> manager >> closing env. Total runtime: ', sTime.delta)
    env.close()
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments')
    parser.add_argument("--mode", type=str, choices=mode_options, default='train', help="luno key id")
    args = parser.parse_args()
    main(args.mode)