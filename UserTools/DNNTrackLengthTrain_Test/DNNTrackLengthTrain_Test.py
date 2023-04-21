# DNNTrackLengthTrain_Test Tool script
# ------------------
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from Tool import *

class DNNTrackLengthTrain_Test(Tool):
    
    # declare member variables here
    inputdatafilename = std.string("")
    weightsfilename = std.string("")
    
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("TrackLengthTrainingInputDataFile", self.inputdatafilename)
        self.m_variables.Get("TrackLengthOutputWeightsFile", self.weightsfilename)
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

thistool = DNNTrackLengthTrain_Test()

def SetToolChainVars(m_data_in, m_variables_in, m_log_in):
    return thistool.SetToolChainVars(m_data_in, m_variables_in, m_log_in)

def Initialise():
    return thistool.Initialise()

def Execute():
    # Set random seed to improve reproducibility
    seed = 150
    np.random.seed(seed)

    print( "--- opening file with input variables!")
    #--- events for training - MC events
    filein = open(str(thistool.inputdatafilename))
    print("evts for training in: ",filein)
    Dataset=np.array(pd.read_csv(filein, index_col=0))
    np.random.shuffle(Dataset)#shuffling the data sample to avoid any bias in the training
    print(Dataset)
    features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)
    print(rest)
    print(features[:,2202])
    print(features[:,2201])
    print(labels)
    #split events in train/test samples:
    num_events, num_pixels = features.shape
    print(num_events, num_pixels)
    np.random.seed(0)
    if(len(labels) % 2==0):
        a=len(labels)/2
    else:
        a=(len(labels)+1)/2
    train_x = features[:a]
    train_y = labels[:a]

    print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

    # Scale data (training set) to 0 mean and unit standard deviation.
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)

    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(25, input_dim=2203, kernel_initializer='normal', activation='relu'))
        model.add(Dense(25, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='relu'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
        return model

    estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=2, verbose=0)

    # checkpoint
    filepath=str(thistool.weightsfilename)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    callbacks_list = [checkpoint]
    # Fit the model
    history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=12, batch_size=1, callbacks=callbacks_list, verbose=0)
    #-----------------------------
    # summarize history for loss
    f, ax2 = plt.subplots(1,1)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Performance')
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(0.,12.)
    ax2.legend(['training loss', 'validation loss'], loc='upper left')
    plt.savefig("keras_train_test.pdf")
    return thistool.Execute()

def Finalise():
    return thistool.Finalise()
