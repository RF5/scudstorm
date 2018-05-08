'''
Scudstorm runner for Tower Defense game engine

Entelect Challenge 2018
Author: Matthew Baas
'''

import time
import subprocess, signal
import os
import json
from shutil import copy2
import shutil
from common.util import write_action

# Config variables
jar_name = 'tower-defence-runner-1.1.0.jar'
config_name = 'config.json'
out_filename = 'env_out.txt'
wrapper_out_filename = 'wrapper_out.txt'
state_name = 'state.json'

class Env():

    def __init__(self, name, debug):
        self.name = name
        self.debug = debug
        self.setup_directory()

        # setting up jar runner
        command = 'python' + self.wrapper_path
        P = subprocess.Popen(command, shell=True)

    def setup_directory(self):
        # creates the dirs responsible for this env, 
        # and moves a copy of the runner and config to that location
        basedir = os.path.dirname(os.path.abspath(__file__)) # now in scudstorm dir
        self.run_path = os.path.join(basedir, 'runs', self.name)
        self.wrapper_path = os.path.join(self.run_path, 'jar_wrapper.py')
        os.makedirs(self.run_path, exist_ok=True)
        jarpath = os.path.join(basedir, jar_name)
        copy2(jarpath, self.run_path)
        config_path = os.path.join(basedir, config_name)
        copy2(config_path, self.run_path)
        wrapper_path = os.path.join(basedir, 'common', 'jar_wrapper.py')
        copy2(wrapper_path, self.run_path)

        self.out_file = os.path.join(self.run_path, out_filename)
        self.in_file = os.path.join(self.run_path, wrapper_out_filename)
        self.state_file = os.path.join(self.run_path, state_name)

        # run path should now have the jar, config and jar wrapper files

    def step(self, action):
        x, y, build = action
        write_action(x, y, build, path=self.run_path)

        with open(self.out_file, 'w') as f:
            # we want start of a new step
            f.write('1')
        obs = None
        while True:
            with open(self.in_file, 'r') as ff:
                k = ff.read()
                k = int(k)
                if k == 1:
                    with open(self.out_file, 'w') as fk:
                        fk.write('0') # do not have another step until we finish
                    # a new turn has just been processed
                    obs = json.load(open(self.state_file,'r'))
                    break
            time.sleep(0.01)


    def close(self):
        # clean up after itself
        print("want to remove: ", self.run_path)
        #shutil.rmtree(self.run_path)
