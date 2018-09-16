'''
tf-logging-util

A quick and easy utility for setting tensorflow's level of output logging

Author: Matthew Baas
'''

import tensorflow as tf
import os

def set_logging_level(level):
    '''
    Sets the tensorflow logging ignore threshold.

    Only the tensorflow log events with a severity/importance level
    higher than the log type indicated by [level] will be shown (on standard output).

    level, a string, may be one of:
    - 'all' : show all tensorflow logging messages
    - 'info' : hide all debug and info messages.
    - 'warn' : hide all debug, info, and warning mesasges
    - 'error' : hide all debug, info, warning, and error messages
    Note: Fatal errors will always be shown.
    '''
    if str(tf.__version__).startswith('1.') == False:
        if level == 'all' or level == 'default':
            tf.logging.set_verbosity(tf.logging.INFO)
        elif level == 'info':
            tf.logging.set_verbosity(tf.logging.WARN)
        elif level == 'warn':
            tf.logging.set_verbosity(tf.logging.ERROR)
        elif level == 'error':
            tf.logging.set_verbosity(tf.logging.FATAL)
        else:
            raise ValueError("The logging level must be one of 'all', 'info', 'warn', or 'error'")
        return

    if type(level) == str:
        if level == 'all' or level == 'default':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        elif level == 'info':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        elif level == 'warn':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        elif level == 'error':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        else:
            raise ValueError("The logging level must be one of 'all', 'info', 'warn', or 'error'")
    else:
        raise TypeError('The logging level given must be a string!')

    return

if __name__ == '__main__':
    print("Running tests...")
    set_logging_level('error')
    set_logging_level('warn')
    set_logging_level('info')
    set_logging_level('all')
    print("Tests complete.")