##### Script to Train DNN for Track Length Reconstruction in the water tank
# bend over backwards for reproducible results
# see https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import numpy
import tensorflow
print('left tensorflow')
import random


# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
numpy.random.seed(0)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
#session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#from tensorflow.keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
#tensorflow.set_random_seed(1234)
#sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
#K.set_session(sess)

import Store
import sys
import glob
import pandas #as pd
import tempfile
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
#import sklearn
print('heading into sklearn, brrrrrrrrrr')
from sklearn import tree
#from sklearn import datasets
#from sklearn import metrics
#from sklearn import model_selection
#from sklearn import preprocessing

#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def Initialise(pyinit):
    print('Hello World')
    return 1
  
def Execute():
    return 1
    
def Finalise():
    return 1
