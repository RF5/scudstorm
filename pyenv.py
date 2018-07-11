'''
Scudstorm runner for Tower Defense game engine

Entelect Challenge 2018
Author: Matthew Baas
'''

import time
import subprocess
import os
import json
import sys
import numpy as np
from shutil import copy2
import shutil
import signal
from common.util import write_prep_action, ControlObject, get_initial_obs
from common.obs_parsing import is_valid_action
from common.metrics import Stopwatch

# Config variables
jar_name = 'tower-defence-runner-2.0.3.jar'
config_name = 'game-runner-config.json'#'config.json'
game_config_name = 'game-config.properties'
wrapper_out_filename = 'wrapper_out.txt'
state_name = 'state.json'
bot_file_name = 'bot.json'
per_step_reward_penalty = -15

binary_step_penalty = -0.01

general_reward_scaling_factor = 0.34

binary_win_reward = 1.0
dense_win_reward = 600

possible_reward_modes = ['dense', 'binary']
reward_mode = possible_reward_modes[0]

keep_log_folder_override = False

class Env():

    def __init__(self, name, debug):
        self.name = name
        self.debug = debug
        self.setup_directory()
        self.score = 0
        self.score_delta = 0
        # setting up jar runner
        self.needs_reset = True
        self.pid = None
        self.done = False
        self.prev_obs = get_initial_obs(1)[0][0]
        self.clock = Stopwatch()
        self.step_num = 0

    def setup_directory(self):
        # creates the dirs responsible for this env, 
        # and moves a copy of the runner and config to that location
        
        print("Setting up file directory for " + self.name + " with pid " + str(os.getpid()))
        basedir = os.path.dirname(os.path.abspath(__file__)) # now in scudstorm dir
        self.run_path = os.path.join(basedir, 'runs', self.name)
        if os.path.isdir(self.run_path):
            shutil.rmtree(self.run_path)
        self.wrapper_path = os.path.join(self.run_path, 'jar_wrapper.py')
        os.makedirs(self.run_path, exist_ok=True)
        jarpath = os.path.join(basedir, jar_name)
        copy2(jarpath, self.run_path)
        config_path = os.path.join(basedir, config_name)
        copy2(config_path, self.run_path)
        wrapper_path = os.path.join(basedir, 'common', 'jar_wrapper.py')
        copy2(wrapper_path, self.run_path)
        botdir = os.path.join(basedir, bot_file_name)
        copy2(botdir, self.run_path)
        copy2(os.path.join(basedir, game_config_name), self.run_path)

        # Copying over reference bot
        self.refbot_path = os.path.join(self.run_path, 'refbot')

        if os.path.isdir(self.refbot_path):
            shutil.rmtree(self.refbot_path)
        refbotdir = os.path.join(basedir, 'refbot')
        shutil.copytree(refbotdir, self.refbot_path)

        self.in_file = os.path.join(self.run_path, wrapper_out_filename)
        self.state_file = os.path.join(self.run_path, state_name)
        self.bot_file = os.path.join(self.run_path, bot_file_name)
        self.proc = None
        self.refenv = RefEnv(self, debug=self.debug)
        
        with open(self.in_file, 'w') as f:
            f.write('0')
        # run path should now have the jar, config and jar wrapper files

    def step(self, action, ref_act):
        #############################
        ## Maintenence on the process
        if self.needs_reset:
            self.reset()

        #######################
        ## Debug messages
        if self.debug:
            with open(os.path.join(self.run_path, 'mylog.txt'), 'a') as f:
                f.write(str(time.time()) + "\t-->Wanting to do op:!!!\t" + str(action) + '\t' + str(ref_act) + '\n')

            with open(os.path.join(self.refbot_path, 'mylog.txt'), 'a') as f:
                f.write(str(time.time()) + "\t-->Wanting to do op:!!!\t" + str(ref_act) + '\n')
        ep_info = {}
        #######################
        ## Writing actions
        x2, y2, build2 = ref_act
        write_prep_action(x2, y2, build2, path=self.refbot_path, debug=self.debug)

        x, y, build = action
        write_prep_action(x, y, build, path=self.run_path, debug=self.debug)

        #######################
        ## Signalling to jar wrappers to begin their running step
        with open(self.in_file, 'w') as f:
            # we want start of a new step
            if self.debug:
                print(">> pyenv {} >> writing 2 to file {}".format(self.name, self.in_file))
            f.write('2')

        with open(self.refenv.in_file, 'w') as f:
            # we want start of a new step
            if self.debug:
                print(">> pyenv {} >> writing 2 to file {}".format(self.refenv.name, self.refenv.in_file))
            f.write('2')

        #######################
        ## Checking if episode ended early
        if self.proc.poll() != None and self.done == True:
            # env ended last step, so reset:
            if self.debug:
                print(">> PYENV ", self.name ," >>  Ended early")
            cntrl_obj = ControlObject('EARLY')
            tp = np.concatenate([np.asarray([cntrl_obj,]), np.asarray([cntrl_obj,])], axis=-1)
            return tp, np.concatenate([np.asarray([cntrl_obj,]), np.asarray([cntrl_obj,])], axis=-1), ep_info

        #######################
        ## Taking step

        # Vars for Env
        obs = None
        should_load_obs = False
        reward = None
        # Vars for ref env
        ref_obs = None
        should_load_obs2 = False
        # Waiting for responses from the jar wrappers
        stopw = Stopwatch()
        failure = False
        while True:
            if should_load_obs == False:
                with open(self.in_file, 'r') as ff:
                    k = ff.read()
                    try:
                        k = int(k)
                    except ValueError:
                        continue
                    if k == 1:
                        #print("just wrote 0 to the ", self.out_file)
                        # a new turn has just been processed
                        should_load_obs = True

            if should_load_obs2 == False:
                with open(self.refenv.in_file, 'r') as ff:
                    k2 = ff.read()
                    try:
                        k2 = int(k2)
                    except ValueError:
                        continue
                    if k2 == 1:
                        #print("just wrote 0 to the ", self.out_file)
                        # a new turn has just been processed
                        should_load_obs2 = True

            if should_load_obs == True and should_load_obs2 == True:
                break
            
            if self.proc.poll() != None and self.done == False:
                #ep ended early.
                if self.debug:
                    print("PYENV: >> GAME ENDING EARLY FOR THE FIRST TIME")
                self.done = True

                valid, reason = is_valid_action(action, self.prev_obs)
                obs = self.load_state()
                self.prev_obs = obs
                
                ep_info['n_steps'] = self.step_num

                if valid == True:
                    ep_info['valid'] = True
                else:
                    ep_info['valid'] = False

                ref_obs = self.refenv.load_state()
                if obs['players'][0]['playerType'] == 'A':
                    a_hp = obs['players'][0]['health']
                    b_hp = obs['players'][1]['health']
                else:
                    a_hp = obs['players'][1]['health']
                    b_hp = obs['players'][0]['health']
                k = np.asarray([obs,])
                u = np.asarray([ref_obs,])
                return_obs = np.concatenate([k, u], axis=-1)
                if reward_mode == 'dense':
                    win_reward = dense_win_reward
                    lose_reward = -1 * dense_win_reward
                else:
                    win_reward = binary_win_reward
                    lose_reward = -1 * binary_win_reward
                if a_hp > b_hp:
                    # player a wins
                    ep_info['winner'] = 'A'
                    return return_obs, np.concatenate([np.asarray([win_reward,]), np.asarray([lose_reward,])], axis=-1), ep_info
                elif a_hp < b_hp:
                    ep_info['winner'] = 'B'
                    return return_obs, np.concatenate([np.asarray([lose_reward,]), np.asarray([win_reward,])], axis=-1), ep_info
                else:
                    ep_info['winner'] = 'TIE'
                    return return_obs, np.concatenate([np.asarray([0.0,]), np.asarray([0.0,])], axis=-1), ep_info

            if stopw.deltaT() > 3:
                # we have waited more than 3s, game clearly ended
                self.needs_reset = True
                failure = True
                print('pyenv: env ' + str(self.name) + ' with pid ' + str(self.pid) + ' encountered error. (', should_load_obs, ',',should_load_obs2, ')' , time.time())
                break

            time.sleep(0.01)
        # TODO: possibly pre-parse obs here and derive a reward from it?
        
        #########################
        ## Loading the obs if their jar's ended properly
        #ref_obs, _ = self.refenv.step(ref_act)
        if should_load_obs:
            valid, reason = is_valid_action(action, self.prev_obs)
            obs = self.load_state()
            self.prev_obs = obs

            if valid == True:
                ep_info['valid'] = True
            else:
                ep_info['valid'] = False
        if should_load_obs2:
            ref_obs = self.refenv.load_state()

        if obs is None and self.debug == True:
            print(">> PY_ENV >> MAIN OBS IS NONE (", self.name, ")")

        if ref_obs is None:
            print(">> PY_ENV >> REF OBS IS NONE. (", self.name, ")")

        if failure == True:
            cntrl_obj = ControlObject('FAILURE')
            tp = np.concatenate([np.asarray([cntrl_obj,]), np.asarray([cntrl_obj,])], axis=-1)
            return tp, np.concatenate([np.asarray([cntrl_obj,]), np.asarray([cntrl_obj,])], axis=-1), ep_info

        # print('-----A------------->', obs['players'][0]['health'])
        # print('-----B------------->', obs['players'][1]['health'])
        self.step_num += 1

        ########################
        ## Forming rewards and packaging the obs into a good numpy form
        if obs is not None:
            # Infer reward:
            #reward = float(obs['players'][0]['score']) - float(obs['players'][1]['score'])
            curS = float(obs['players'][0]['score']) * general_reward_scaling_factor
            self.score_delta = curS - self.score
            reward = self.score_delta + per_step_reward_penalty
            self.score = curS

        if ref_obs is not None:
            curS2 = float(ref_obs['players'][0]['score']) * general_reward_scaling_factor
            self.refenv.score_delta = curS2 - self.refenv.score
            ref_reward = self.refenv.score_delta + per_step_reward_penalty
            self.refenv.score = curS2

        k = np.asarray([obs,])
        u = np.asarray([ref_obs,])
        return_obs = np.concatenate([k, u], axis=-1)
        if reward_mode == 'dense':
            return return_obs, np.concatenate([np.asarray([reward,]), np.asarray([ref_reward,])], axis=-1), ep_info
        elif reward_mode == 'binary':
            return return_obs, np.concatenate([np.asarray([binary_step_penalty,]), np.asarray([binary_step_penalty,])], axis=-1), ep_info


    def load_state(self):
        '''
        Gets the current Game State json file.
        '''
        while os.path.isfile(self.state_file) == False:
            if self.debug:
               print(">> PYENV >> waiting for state file  ", self.state_file, ' to appear')
            time.sleep(0.01)

        flag = False
        while flag == False:
            try:
                k = json.load(open(self.state_file,'r'))
                flag = True
                break
            except json.decoder.JSONDecodeError as e:
                k = None
                if self.debug:
                    print(">> PYENV >> Failed to decode json state! Got error ", e)
                time.sleep(0.01)

        return k

    def get_obs(self):
        this_obs = self.load_state()
        refbot_obs = self.refenv.load_state()
        x = np.asarray([this_obs,])
        y = np.asarray([refbot_obs,])

        return np.concatenate([x, y], axis=-1)

    def reset(self):
        self.step_num = 0
        if self.debug:
            with open(os.path.join(self.run_path, 'mylog.txt'), 'a') as f:
                f.write(str(time.time()) + "\t-->RESETTING!!!\n")

            with open(os.path.join(self.refbot_path, 'mylog.txt'), 'a') as f:
                f.write(str(time.time()) + "\t-->RESETTING!!!\n")

        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
        self.needs_reset = False
        self.done = False
        time.sleep(0.01)
        # trying to kill jar wrapper of this env
        pid_file = os.path.join(self.run_path, 'wrapper_pid.txt')
        if os.path.isfile(pid_file):
            flag = False
            while flag == False:
                with open(pid_file, 'r') as f:
                    try:
                        wrapper_pid = int(f.read())
                    except ValueError:
                        continue
                    if wrapper_pid == 0:
                        flag = True
                        return None
                    else:
                        flag = True
                        try:
                            os.kill(wrapper_pid, signal.SIGTERM)
                        except (PermissionError, ProcessLookupError) as e:
                            if self.debug:
                                print(">> PYENV ", self.name, " >> Attempted to close wrapper pid ", wrapper_pid, " but got ERROR ", e)
                        break
        else:
            if self.debug:
                print(">> PYENV >> Attempted to close wrapper pid but the wrapper pid file was not found ")
        ## Trying to prevent reset bugs from propping up
        # if os.path.isdir(self.refbot_path):
        #     shutil.rmtree(self.refbot_path)
        # refbotdir = os.path.join(basedir, 'refbot')
        # shutil.copytree(refbotdir, self.refbot_path)

        ## Trying to kill jar wrapper of ref env
        refpid_file = os.path.join(self.refbot_path, 'wrapper_pid.txt')
        if os.path.isfile(refpid_file):
            flag = False
            while flag == False:
                with open(refpid_file, 'r') as f:
                    try:
                        wrapper_pid2 = int(f.read())
                    except ValueError:
                        continue
                    if wrapper_pid2 == 0:
                        flag = True
                        return None
                    else:
                        flag = True
                        try:
                            os.kill(wrapper_pid2, signal.SIGTERM)
                        except (PermissionError, ProcessLookupError) as e:
                            if self.debug:
                                print(">> PYENV ", self.name, " >> Attempted to close refbot wrapper pid ", wrapper_pid2, " but got ERROR ", e)
        else:
            if self.debug:
                print(">> PYENV >> Attempted to close refbot wrapper pid but the wrapper pid file was not found ")
        time.sleep(0.01)
        
        #######################
        ## Flushing matchlogs folder if env alive for over 1h
        if self.clock.deltaT() >= 1800:
            print(">> PYENV {} >> Env alive for over half an hour, flushing (deleting) matchlogs folder".format(self.name))
            self.cleanup()
            self.clock.reset()
            print("Cleand.")

        command = 'java -jar ' + os.path.join(self.run_path, jar_name)

        if sys.platform == "win32":
            she = False
        else:
            she = True

        if self.debug:
            self.proc = subprocess.Popen(command, shell=she , stdout=subprocess.PIPE, cwd=self.run_path)
            print("Opened process: ", str(command), " with pid ", self.proc.pid)
        else:
            self.proc = subprocess.Popen(command, shell=she, stdout=subprocess.DEVNULL, cwd=self.run_path)
        
        self.pid = self.proc.pid
        time.sleep(0.01)

        return True

    def close(self):
        if self.debug:
            print("Closing env ", self.name)
        # clean up after itself
        
        if self.pid is not None:
            self.needs_reset = True
            self.proc.terminate()
            self.proc.wait()
        else:
            return None

        time.sleep(0.1)
        pid_file = os.path.join(self.run_path, 'wrapper_pid.txt')
        if os.path.isfile(pid_file):
            flag = False
            while flag == False:
                with open(pid_file, 'r') as f:
                    try:
                        wrapper_pid = int(f.read())
                    except ValueError:
                        continue
                    if wrapper_pid == 0:
                        flag = True
                        return None
                    else:
                        flag = True
                        try:
                            os.kill(wrapper_pid, signal.SIGTERM)
                        except (PermissionError, ProcessLookupError) as e:
                            if self.debug:
                                print(">> PYENV ", self.name, " >> Attempted to close wrapper pid ", wrapper_pid, " but got ERROR ", e)
                        break
        else:
            print(">> PYENV >> Attempted to close wrapper pid but the wrapper pid file was not found ")
        time.sleep(0.1)

        refpid_file = os.path.join(self.refbot_path, 'wrapper_pid.txt')
        if os.path.isfile(refpid_file):
            flag = False
            while flag == False:
                with open(refpid_file, 'r') as f:
                    try:
                        wrapper_pid2 = int(f.read())
                    except ValueError:
                        continue
                    if wrapper_pid2 == 0:
                        flag = True
                        return None
                    else:
                        flag = True
                        try:
                            os.kill(wrapper_pid2, signal.SIGTERM)
                        except (PermissionError, ProcessLookupError) as e:
                            if self.debug:
                                print(">> PYENV ", self.name, " >> Attempted to close refbot wrapper pid ", wrapper_pid2, " but got ERROR ", e)
        else:
            if self.debug:
                print(">> PYENV >> Attempted to close refbot wrapper pid but the wrapper pid file was not found ")
        time.sleep(0.1)

        self.pid = None
        return True
        
    def cleanup(self):
        log_path = os.path.join(self.run_path, 'matchlogs')

        if self.debug:
            print("Removing folder: ", log_path)
        try:
            if keep_log_folder_override == False:
                shutil.rmtree(log_path)
            else:
                print(">> PYENV >> OVERRIDE - Keeping log files.")
            time.sleep(0.1)
        except Exception:
            print(">> PYENV >> Exception occured while removing matchlogs folder")

class RefEnv():
    def __init__(self, env, debug=False):
        self.ref_path = env.refbot_path
        self.name = env.name + '_refbot'
        self.debug = debug
        self.in_file = os.path.join(self.ref_path, wrapper_out_filename)
        self.state_file = os.path.join(self.ref_path, state_name)
        self.bot_file = os.path.join(self.ref_path, bot_file_name)
        self.score = 0
        self.score_delta = 0

    def step(self, action):
        raise NotImplementedError
        # return obs, 0

    def load_state(self):
        '''
        Gets the current Game State json file.
        '''
        while os.path.isfile(self.state_file) == False:
            if self.debug:
                print(">> PYENV REFBOT >> waiting for state file  ", self.state_file, ' to appear')
            time.sleep(0.01)

        flag = False
        while flag == False:
            try:
                k = json.load(open(self.state_file,'r'))
                flag = True
                break
            except json.decoder.JSONDecodeError as e:
                k = None
                if self.debug:
                    print(">> REF ENV >> Failed to decode json state! Got error ", e)
                time.sleep(0.01)

        return k