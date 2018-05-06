'''
Scudstorm runner for Tower Defense game engine

Entelect Challenge 2018
Author: Matthew Baas
'''

import time
import multiprocessing
from storm import Storm

def train_runner():
    with open('running_state.txt', 'w') as f:
        f.write('1')

    flag = True
    while flag:
        with open('running_state.txt', 'r') as f:
            k = int(f.read())
        if k != 0:
            time.sleep(0.1)
        else:
            flag = False
            break

def entry():
    s = Storm()

if __name__ == '__main__':
    with open('running_state.txt', 'r') as f:
        k = int(f.read())
    if k == 3:
        # start storm in a subprocess
        with open('running_state.txt', 'w') as f:
            f.write('0')
        #p = multiprocessing.Process(target=entry, daemon=False)
        #p.start()

    train_runner()
