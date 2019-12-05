"""
@title
@description
"""
import os

'''
have to set environment variable before importing tensorflow

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
TENSORFLOW_LOG_LEVEL = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TENSORFLOW_LOG_LEVEL
