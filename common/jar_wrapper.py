'''
Scudstorm java wrapper

Entelect Challenge 2018
Author: Matthew Baas
'''

import os
import time
import sys
from shutil import copy2
import traceback

debug = False

def fileLog(msg):
    if debug:
        with open('mylog.txt', 'a') as f:
            f.write(str(time.time()) + ':\t' + str(msg) + "\n")

def main():
    try:
        # Config (generics)
        wrapper_out_filename = 'wrapper_out.txt'

        this_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__))) # inside our running dir
        command_name = os.path.join(this_dir, 'command2.txt')
        proper_name = os.path.join(this_dir, 'command.txt')
        pid_file = os.path.join(this_dir, 'wrapper_pid.txt')
        wrapper_path = os.path.join(this_dir, wrapper_out_filename)

        with open(pid_file, 'w') as ff:
            fileLog(os.getpid())
            ff.write(str(os.getpid()) + "\n")

        meme = False
        while meme == False:
            while os.path.isfile(wrapper_path) == False:
                time.sleep(0.01)

            with open(wrapper_path, 'r') as f:
                k = f.read()
                fileLog("we got k as " + str(k))
                if k == "2":
                    flag = True
                    break
                time.sleep(0.01)

        flag = False
        while flag == False:
            while os.path.isfile(command_name) == False:
                time.sleep(0.01)

            with open(command_name, 'r') as f:
                meme = f.read()
                fileLog("we got meme as " + str(meme))
                if meme is not None and meme != "":
                    flag = True
                    break
                time.sleep(0.01)
                    
        if meme == "NO_OP":
            outfl = open('command.txt','w')
            outfl.write("")
            outfl.close()
            #with open(command_name, 'w') as f:
            #    f.write(" ")
        else:
            outfl = open('command.txt','w')
            outfl.write(meme)
            outfl.close()

        #copy2(command_name, proper_name)
        os.remove(command_name)
        #os.rename(command_name, proper_name)
        
        fileLog("Found it!! Exiting to next round!")
        with open(wrapper_path, 'w') as f:
            f.write('1')
        fileLog("Wrote 1 to wrapper file! We should not end on this")
        sys.exit(0)
    except Exception as err:
        try:
            exc_info = sys.exc_info()

        finally:
            k = traceback.format_exception(*exc_info)
            with open('mylog2.txt', 'w') as f:
                for line in k:
                    f.write(line + "\n")
            del exc_info

if __name__ == '__main__':
    main()