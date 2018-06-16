'''
Scudstorm runner for Tower Defense game engine

Entelect Challenge 2018
Author: Matthew Baas
'''

import time
import subprocess
import os
import json
import numpy as np
from shutil import copy2
import shutil
import signal
from common.util import write_prep_action
from common.metrics import Stopwatch

# Config variables
jar_name = 'tower-defence-runner-1.1.0.jar'
config_name = 'config.json'
wrapper_out_filename = 'wrapper_out.txt'
state_name = 'state.json'
bot_file_name = 'bot.json'

class Env():

    def __init__(self, name, debug):
        self.name = name
        self.debug = debug
        self.setup_directory()
        self.score = 0
        # setting up jar runner
        self.needs_reset = True
        self.pid = None

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

        # Copying over starter bot -- replace this in future:
        # if os.path.isdir(os.path.join(self.run_path, 'python3')) == False:
        #     refbotdir = os.path.join(basedir, 'python3')
        #     shutil.copytree(refbotdir, os.path.join(self.run_path, 'python3'))
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
        if self.needs_reset:
            self.reset()
        x, y, build = action
        
        write_prep_action(x, y, build, path=self.run_path, debug=self.debug)

        with open(self.in_file, 'w') as f:
            # we want start of a new step
            if self.debug:
                print(">> pyenv {} >> writing 2 to file {}".format(self.name, self.in_file))
            f.write('2')
        obs = None
        reward = None
        stopw = Stopwatch()
        while True:
            with open(self.in_file, 'r') as ff:
                k = ff.read()
                try:
                    k = int(k)
                except ValueError:
                    continue
                if k == 1:
                    #print("just wrote 0 to the ", self.out_file)
                    # a new turn has just been processed
                    obs = self.load_state()
                    break
            
            if stopw.deltaT() > 10:
                # we have waited more than 3s, game clearly ended
                self.needs_reset = True
                #if self.debug:
                print('pyenv: env ' + str(self.name) + ' with pid ' + str(self.pid) + ' needs reset.')
                break

            time.sleep(0.01)
        # TODO: possibly pre-parse obs here and derive a reward from it?
        
        if obs is None and self.debug == True:
            print(">> PY_ENV >> MAIN OBS IS NONE")

        if obs is not None:
            # Infer reward:
            #reward = float(obs['players'][0]['score']) - float(obs['players'][1]['score'])
            curS = float(obs['players'][0]['score'])
            reward = curS - self.score
            self.score = curS

        ref_obs, _ = self.refenv.step(ref_act)

        k = np.asarray([obs,])
        u = np.asarray([ref_obs,])
        return_obs = np.concatenate([k, u], axis=-1)
        return return_obs, np.concatenate([np.asarray([reward,]), np.asarray([0.0,])], axis=-1)

    def load_state(self):
        '''
        Gets the current Game State json file.
        '''
        while os.path.isfile(self.state_file) == False:
            if self.debug:
                print(">> PYENV >> waiting for state file  ", self.state_file, ' to appear')
            time.sleep(0.01)
        try:
            k = json.load(open(self.state_file,'r'))
        except json.decoder.JSONDecodeError as e:
            k = None
            print("Failed to decode json state! Got error ", e)

        return k

    def get_obs(self):
        this_obs = self.load_state()
        refbot_obs = self.refenv.load_state()
        x = np.asarray([this_obs,])
        y = np.asarray([refbot_obs,])

        return np.concatenate([x, y], axis=-1)

    def reset(self):
        if self.proc is not None:
            self.proc.kill()
        self.needs_reset = False

        pid_file = os.path.join(self.run_path, 'wrapper_pid.txt')
        if os.path.isfile(pid_file):
            with open(pid_file, 'r') as f:
                wrapper_pid = int(f.read())
                if wrapper_pid == 0:
                    return None
                else:
                    try:
                        os.kill(wrapper_pid, signal.SIGTERM)
                    except PermissionError as e:
                        if self.debug:
                            print(">> PYENV ", self.name, " >> Attempted to close wrapper pid ", wrapper_pid, " but got ERROR ", e)
        else:
            if self.debug:
                print(">> PYENV >> Attempted to close wrapper pid but the wrapper pid file was not found ")
        time.sleep(0.1)

        command = 'java -jar ' + os.path.join(self.run_path, jar_name)
        if self.debug:
            self.proc = subprocess.Popen(command, cwd=self.run_path)
            print("Opened process: ", str(command), " with pid ", self.proc.pid)
        else:
            self.proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, cwd=self.run_path)
        
        self.pid = self.proc.pid
        time.sleep(0.2)

        return True

    def close(self):
        if self.debug:
            print("Closing env ", self.name)
        # clean up after itself
        
        if self.pid is not None:
            self.needs_reset = True
            self.proc.kill()
        else:
            return None

        time.sleep(0.1)
        pid_file = os.path.join(self.run_path, 'wrapper_pid.txt')
        if os.path.isfile(pid_file):
            with open(pid_file, 'r') as f:
                wrapper_pid = int(f.read())
                if wrapper_pid == 0:
                    return None
                else:
                    try:
                        os.kill(wrapper_pid, signal.SIGTERM)
                    except PermissionError as e:
                        print(">> PYENV >> Attempted to close wrapper pid ", wrapper_pid, " but got ERROR ", e)
        else:
            print(">> PYENV >> Attempted to close wrapper pid but the wrapper pid file was not found ")
        time.sleep(0.1)
        # pid_file2 = os.path.join(self.refenv.ref_path, 'wrapper_pid.txt')
        # with open(pid_file2, 'r') as f:
        #     wrapper_pid2 = int(f.read())
        #     if wrapper_pid2 == 0:
        #         return None
        #     else:
        #         os.kill(wrapper_pid2, signal.SIGTERM)

        self.pid = None
        return True
        
    def cleanup(self):
        log_path = os.path.join(self.run_path, 'matchlogs')

        if self.debug:
            print("Removing folder: ", log_path)
        try:
            shutil.rmtree(log_path)
        except Exception:
            pass

class RefEnv():
    def __init__(self, env, debug=False):
        self.ref_path = env.refbot_path
        self.name = env.name + '_refbot'
        self.debug = debug
        self.in_file = os.path.join(self.ref_path, wrapper_out_filename)
        self.state_file = os.path.join(self.ref_path, state_name)
        self.bot_file = os.path.join(self.ref_path, bot_file_name)

    def step(self, action):
        x, y, build = action

        write_prep_action(x, y, build, path=self.ref_path, debug=self.debug)

        with open(self.in_file, 'w') as f:
            # we want start of a new step
            if self.debug:
                print(">> pyenv {} >> writing 2 to file {}".format(self.name, self.in_file))
            f.write('2')
        obs = None
        #reward = None
        stopw = Stopwatch()
        while True:
            with open(self.in_file, 'r') as ff:
                k = ff.read()
                try:
                    k = int(k)
                except ValueError:
                    continue
                if k == 1:
                    #print("just wrote 0 to the ", self.out_file)
                    # a new turn has just been processed
                    obs = self.load_state()
                    break
            
            if stopw.deltaT() > 10:
                # we have waited more than 3s, game clearly ended
                self.needs_reset = True
                #if self.debug:
                print('pyenv: something very bad happened in refbot ', self.name, '. Took more than 10s to respond...')
                break

            time.sleep(0.01)
        # TODO: possibly pre-parse obs here and derive a reward from it?
        
        # if obs is not None:
        #     # Infer reward:
        #     #reward = float(obs['players'][0]['score']) - float(obs['players'][1]['score'])
        #     curS = float(obs['players'][0]['score'])
        #     reward = curS - self.score
        #     self.score = curS
        if obs is None:
            print(">> PY_ENV >> REF OBS IS NONE")

        return obs, 0

    def load_state(self):
        '''
        Gets the current Game State json file.
        '''
        while os.path.isfile(self.state_file) == False:
            if self.debug:
                print(">> PYENV REFBOT >> waiting for state file  ", self.state_file, ' to appear')
            time.sleep(0.01)

        try:
            k = json.load(open(self.state_file,'r'))
        except json.decoder.JSONDecodeError as e:
            k = None
            print("Failed to decode json state! Got error ", e)

        return k