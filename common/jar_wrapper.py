'''
Scudstorm java wrapper

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import time

# Config (generics)
in_filename = 'env_out.txt'
wrapper_out_filename = 'wrapper_out.txt'

def main():
    this_dir = os.path.dirname(os.path.abspath(__file__)) # inside our running dir
    print("our inner dir = ", this_dir)
    with open(wrapper_out_filename, 'w') as f:
        f.write('1')

    while True:
        time.sleep(0.1)
        try:
            with open(in_filename, 'r') as ff:
                k = ff.read()
                k = int(k)
                if k == 1:
                    with open(wrapper_out_filename, 'w') as fff:
                        fff.write('0') # waiting for a new turn
                    # they want start of a new step, so end this and run a turn
                    break
        except:
            pass

if __name__ == '__main__':
    main()