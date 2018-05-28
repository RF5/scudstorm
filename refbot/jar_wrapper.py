'''
Scudstorm java wrapper

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import time
import os
from shutil import copy2

def fileLog(msg):
    with open('mylog.txt', 'a') as f:
        f.write(str(msg) + "\n")

def main():
    # Config (generics)
    wrapper_out_filename = 'wrapper_out.txt'

    this_dir = os.path.dirname(os.path.abspath(__file__)) # inside our running dir
    
    command_name = os.path.join(this_dir, 'command2.txt')
    proper_name = os.path.join(this_dir, 'command.txt')

    meme = False
    while meme == False:
        with open(os.path.join(this_dir, wrapper_out_filename), 'r') as f:
            k = int(f.read())
            
        if k == 2:
            fileLog("k = " + str(k))
            meme = True
            break
        time.sleep(0.05)
            

    fileLog("our inner dir = " + str(this_dir))
    with open(os.path.join(this_dir, wrapper_out_filename), 'w') as f:
        f.write('1')

    fileLog("help me! command_name = " + str(command_name))
    while os.path.isfile(command_name) == False:
        time.sleep(0.05)
    with open(command_name, 'r') as f:
        fileLog(f.read())

    copy2(command_name, proper_name)
    #os.remove(command_name)
    #os.rename(command_name, proper_name)
    
    fileLog("Found it!! Exiting to next round!")

    with open(os.path.join(this_dir, wrapper_out_filename), 'w') as f:
        f.write('0')
    fileLog("Wrote 0 to wrapper file! We should not end on this")
    # while True:
    #     try:
    #         with open(os.path.join(this_dir, in_filename), 'r') as ff:
    #             k = ff.read()
    #             k = int(k)
    #             if k == 1:
    #                 fileLog("Saw 1, writing 0")
    #                 with open(os.path.join(this_dir, wrapper_out_filename), 'w') as fff:
    #                     fff.write('0') # waiting for a new turn
    #                 # they want start of a new step, so end this and run a turn
    #                 fileLog("just wrote 0, now breaking")
    #                 break
    #     except:
    #         pass
    #     time.sleep(0.02)

if __name__ == '__main__':
    main()