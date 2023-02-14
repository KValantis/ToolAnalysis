##### Script To Validate DNN for Track Length Reconstruction in the water tank
# bend over backwards for reproducible results
# see https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy
import tensorflow
import random
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
numpy.random.seed(0) #42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
#session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tensorflow.config.threading.set_intra_op_parallelism_threads(1)
tensorflow.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow.keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tensorflow.random.set_seed(1234)
#sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
#K.set_session(sess)

import sys
import glob
#import numpy #as np
import pandas #as pd
#import tensorflow #as tf
import tempfile
#import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot #as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K
import pprint
#import Store and other modules
from Tool import *

class DNNTrackLengthPredict(Tool):
    
    # declare member variables here
    inputdatafilename = std.string("")
    weightsfilename = std.string("")
    predictionsfilename = std.string("")
        
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("TrackLengthTestingDataFile", self.inputdatafilename)
        self.m_variables.Get("TrackLengthWeightsFile", self.weightsfilename)
        self.m_variables.Get("TrackLengthPredictionsDataFile", self.predictionsfilename)
        return 1
    
    def Execute(self):
        self.m_log.Log(__file__+" Executing", self.v_debug, self.m_verbosity)
        return 1
    
    def Finalise(self):
        self.m_log.Log(__file__+" Finalising", self.v_debug, self.m_verbosity)
        return 1

###################
# ↓ Boilerplate ↓ #
###################

thistool = DNNTrackLengthPredict()

def SetToolChainVars(m_data_in, m_variables_in, m_log_in):
    return thistool.SetToolChainVars(m_data_in, m_variables_in, m_log_in)

def Initialise():
    return thistool.Initialise()

def Execute():
    # Load Data
    #-----------------------------
    #print( "--- loading input variables from store!")
    print("opening testing data file ", thistool.inputdatafilename)
    testfile = open(str(thistool.inputdatafilename))
    print("evts for testing in: ", testfile)
    # read into a pandas structure
    print("reading file with pandas")
    testfiledata = pandas.read_csv(testfile, sep=',', header=0)
    print(testfiledata.head())
    print("closing file")
    testfile.close()
    # convert to 2D numpy array
    print("converting to numpy array")
    TestingDataset = numpy.array(testfiledata)
    # split the numpy array up into sub-arrays
    testfeatures, testlambdamax, testlabels, testrest = numpy.split(TestingDataset,[2203,2204,2205],axis=1)
    # print info
    print( "lambdamax ", testlambdamax[:2], testlabels[:2])
    #print(testfeatures[0])
    num_events, num_pixels = testfeatures.shape
    print(num_events, num_pixels)
    # Preprocess data and load model
    #-----------------------------
    
    # rename variables for obfuscation
    test_x = testfeatures
    test_y = testlabels
    print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

    # Scale data to 0 mean and unit standard deviation.
    print("scaling to 0 mean and unit std-dev")
    scaler = preprocessing.StandardScaler()
    x_transformed = scaler.fit_transform(test_x)  # are we ok doing fit_transform on test data?
    # scale the features
    features_transformed = scaler.transform(testfeatures)
    
    # define keras model, loading weight from weights file
    print("defining the model")
    model = Sequential()
    print("adding layers")
    model.add(Dense(25, input_dim=2203, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))

    # load weights
    print("loading weights from file ", thistool.weightsfilename)
    model.load_weights(str(thistool.weightsfilename))

    # Compile model
    print("compiling model")
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    print("Created model and loaded weights from file", thistool.weightsfilename)

    # Score accuracy / Make predictions
    #----------------------------------
    print('predicting...')
    y_predicted = model.predict(x_transformed)
    
    # estimate accuracy on dataset using loaded weights
    print("evalulating model on test")
    scores = model.evaluate(test_x, test_y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # Score with sklearn.
    print("scoring sk mse")
    score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
    print('MSE (sklearn): {0:f}'.format(score_sklearn))

    # Write to the output csv file
    #------------------------------
    if (thistool.predictionsfilename is None) or ( thistool.predictionsfilename == ''):
        # no output files today
        print("no output file specified, not writing to file")
    
    print("writing predictions to output file "+thistool.predictionsfilename)
    # build a dataframe from the true and predicted track lengths
    print("building output dataframe")
    outputdataarray = numpy.concatenate((test_y, y_predicted),axis=1)
    outputdataframe=pandas.DataFrame(outputdataarray, columns=['TrueTrackLengthInWater','DNNRecoLength'])
    
    # append as additional columns to the input dataframe
    print("inserting True and Predicted lengths into file data")
    testfiledata.insert(2217, 'TrueTrackLengthInWater', outputdataframe['TrueTrackLengthInWater'].values, allow_duplicates="True")
    testfiledata.insert(2218, 'DNNRecoLength', outputdataframe['DNNRecoLength'].values, allow_duplicates="True")
    
    # write to csv file
    print("writing all data to "+thistool.predictionsfilename)
    testfiledata.to_csv(str(thistool.predictionsfilename), float_format = '%.3f')

    print("clearing session")
    K.clear_session()
    
    print("done; returning") 
    return thistool.Execute()

def Finalise():
    return thistool.Finalise()
