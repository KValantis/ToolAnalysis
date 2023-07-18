##### Script To Validate DNN for Track Length Reconstruction in the water tank
# bend over backwards for reproducible results
# see https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy
import tensorflow
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
import numpy as np
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
import ROOT
#import Store and other modules
from Tool import *

class DNNTrackLengthPredict(Tool):

    # declare member variables here
    weightsfilename = std.string("")
    ScalingVarsBoostStorepathname = std.string("")
        
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("TrackLengthWeightsFile", self.weightsfilename)
        self.m_variables.Get("ScalingVarsBoostStoreFile", self.ScalingVarsBoostStorepathname)
        return 1
        
    def Execute(self):
        self.m_log.Log(__file__+" Executing", self.v_debug, self.m_verbosity)
        # Load Data from EnergyReco Boost Store
        #-----------------------------
        EnergyRecoBoostStore=cppyy.gbl.BoostStore(True, 2)#define the energy boost store class object to load the variables for the DNN training
        EnergyRecoBoostStore = self.m_data.Stores.at("EnergyReco")
        #Retrieve the required variables from the Store
        EnergyRecoBoostStore.Print(False)
        get_ok = EnergyRecoBoostStore.Has("MaxTotalHitsToDNN")
        MaxTotalHitsToDNN=ctypes.c_int(0)
        if not get_ok:
            print("There is no entry in Energy Reco boost store.")
            return 1
        if get_ok:
            print("EnergyRecoBoostStore has entry MaxTotalHitsToDNN: ",get_ok)
            print("type of MaxTotalHitsToDNN entry is :",EnergyRecoBoostStore.Type("MaxTotalHitsToDNN"))
            print("Getting MaxTotalHitsToDNN from EnergyRecoBoostStore")#we are going to use it to instantiate the lambda and digit times vectors
            EnergyRecoBoostStore.Get("MaxTotalHitsToDNN",MaxTotalHitsToDNN)
        print("MaxTotalHitsToDNN is: ", MaxTotalHitsToDNN.value)
        ok = EnergyRecoBoostStore.Has("lambda_vec")
        lambda_vector=std.vector['double'](range(MaxTotalHitsToDNN.value))
        if ok:
            print("EnergyRecoBoostStore has entry lambda_vec: ",ok)
            print("type of lambda_vec entry is :", EnergyRecoBoostStore.Type("lambda_vec"))
            print("Getting lambda_vec from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("lambda_vec", lambda_vector)
        print("The lambda for the first digit is: ", lambda_vector.at(0))
        ok = EnergyRecoBoostStore.Has("digit_ts_vec")
        digit_ts_vector=std.vector['double'](range(MaxTotalHitsToDNN.value))
        if ok:
            print("EnergyRecoBoostStore has entry digit_ts_vec: ",ok)
            print("type of digit_ts_vec entry is :",EnergyRecoBoostStore.Type("digit_ts_vec"))
            print("Getting digit_ts_vec from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("digit_ts_vec", digit_ts_vector)
        print("The digit time for the first digit is: ", digit_ts_vector.at(0))
        ok = EnergyRecoBoostStore.Has("lambda_max")
        lambda_max=ctypes.c_double(0)
        if ok:
            print("EnergyRecoBoostStore has entry lambda_max: ",ok)
            print("type of lambda_max entry is :",EnergyRecoBoostStore.Type("lambda_max"))
            print("Getting lambda_max from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("lambda_max",lambda_max)
        print("Lambda_max is: ", lambda_max.value)
        ok = EnergyRecoBoostStore.Has("num_pmt_hits")
        num_pmt_hits=ctypes.c_int(0)
        if ok:
            print("EnergyRecoBoostStore has entry num_pmt_hits: ",ok)
            print("type of num_pmt_hits entry is :",EnergyRecoBoostStore.Type("num_pmt_hits"))
            print("Getting num_pmt_hits from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("num_pmt_hits",num_pmt_hits)
        print("Number of pmt hits is: ", num_pmt_hits.value)
        ok = EnergyRecoBoostStore.Has("num_lappd_hits")
        num_lappd_hits=ctypes.c_int(0)
        if ok:
            print("EnergyRecoBoostStore has entry num_lappd_hits: ",ok)
            print("type of num_lappd_hits entry is :",EnergyRecoBoostStore.Type("num_lappd_hits"))
            print("Getting num_lappd_hits from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("num_lappd_hits",num_lappd_hits)
        print("Number of lappd hits is: ", num_lappd_hits.value)
        ok = EnergyRecoBoostStore.Has("TrueTrackLengthInWater")
        TrueTrackLengthInWater=ctypes.c_float(0)
        if ok:
            print("EnergyRecoBoostStore has entry TrueTrackLengthInWater: ",ok)
            print("type of TrueTrackLengthInWater entry is :",EnergyRecoBoostStore.Type("TrueTrackLengthInWater"))
            print("Getting TrueTrackLengthInWater from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("TrueTrackLengthInWater",TrueTrackLengthInWater)
        print("TrueTrackLengthInWater is: ", TrueTrackLengthInWater.value)
            
        #Create features and labels and preprocess data for the model
        features_list=[]
        for i in range(lambda_vector.size()):
            features_list.append(lambda_vector.at(i))
        for j in range(digit_ts_vector.size()):
            features_list.append(digit_ts_vector.at(j))
        features_list.append(lambda_max.value)
        features_list.append(num_pmt_hits.value)
        features_list.append(num_lappd_hits.value)
        #make the features and labels numpy array
        features=np.array(features_list)
        labels=np.array([TrueTrackLengthInWater.value])
        
        print(features)
        print(features.shape)
        print(labels)
        print(labels.shape)

        #Load scaling parameters
        ScalingVarsStore=cppyy.gbl.BoostStore(True, 0)
        ok=ScalingVarsStore.Initialise(self.ScalingVarsBoostStorepathname)
        features_mean_values_vec=std.vector['double'](range(len(features)))
        features_std_values_vec=std.vector['double'](range(len(features)))
        ScalingVarsStore.Get("features_mean_values",features_mean_values_vec)
        ScalingVarsStore.Get("features_std_values",features_std_values_vec)

        #Transform data
        test_x=[]

        for i in range(len(features)):
            test_x.append((features[i]-features_mean_values_vec.at(i))/features_std_values_vec.at(i))
        test_X=np.array(test_x).reshape(1,2203)
        
        print("defining the model")
        model = Sequential()
        print("adding layers")
        model.add(Dense(25, input_dim=2203, kernel_initializer='normal', activation='relu'))
        model.add(Dense(25, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='relu'))

        # load weights                                                                                                                                                                                                 
        print("loading weights from file ", self.weightsfilename)
        model.load_weights(str(self.weightsfilename))

        # Compile model                                                                                                                                                                                                
        print("compiling model")
        model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
        print("Created model and loaded weights from file", self.weightsfilename)

        # Score accuracy / Make predictions                                                                                                                                                                            
        #----------------------------------                                                                                                                                                                            
        print('predicting...')
        y_predicted = model.predict(test_X)
        print(y_predicted.shape)

        # estimate accuracy on dataset using loaded weights                                                                                                                                                            
        print("evalulating model on test")
        scores = model.evaluate(test_X, labels, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        # Score with sklearn.
        print("scoring sk mse")
        score_sklearn = metrics.mean_squared_error(y_predicted, labels)
        print("True:",y_predicted,"Reco:",labels)
        print('MSE (sklearn): {0:f}'.format(score_sklearn))
        
        print("clearing session")
        K.clear_session()

        #Set the DNNRecoLength in the EnergyReco boost store for next tools
        DNNRecoLength=ctypes.c_double(y_predicted[0])
        EnergyRecoBoostStore.Set("DNNRecoLength", DNNRecoLength)
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
    return thistool.Execute()

def Finalise():
    return thistool.Finalise()
