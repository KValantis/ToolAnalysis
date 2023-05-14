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
    fUseStore=ctypes.c_bool()
        
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("TrackLengthWeightsFile", self.weightsfilename)
        self.m_variables.Get("UseStore", self.fUseStore)
        if(not self.fUseStore):
          self.m_variables.Get("TrackLengthPredictionsInputDataFile", self.inputdatafilename)
          self.m_variables.Get("TrackLengthPredictionsDataFile", self.predictionsfilename)
        return 1
    
    def Execute(self):
        self.m_log.Log(__file__+" Executing", self.v_debug, self.m_verbosity)
        # Load Data
        #-----------------------------
        #Check if we are going to load the variables from Store or not.
        if(self.fUseStore):
            #print( "--- loading input variables from store!")
            #Get the Energy Reco Boost Store from DataModel
            EnergyRecoBoostStore = self.m_data.Stores.at("EnergyReco")
            EnergyRecoBoostStore.Print(False)
            ok = EnergyRecoBoostStore.Has("MaxTotalHitsToDNN")
            if(ok):
             print("EnergyRecoBoostStore has entry MaxTotalHitsToDNN: ",ok)
             print("type of MaxTotalHitsToDNN entry is :",EnergyRecoBoostStore.Type("MaxTotalHitsToDNN"))
             print("get MaxTotalHitsToDNN from EnergyRecoBoostStore")#we are going to use it to instantiate the lambda and digit times vectors
             MaxTotalHitsToDNN=ctypes.c_int()
             # get the contents of the Energy Reco Store
             EnergyRecoBoostStore.Get("MaxTotalHitsToDNN",MaxTotalHitsToDNN)
             print("MaxTotalHitsToDNN is: ", MaxTotalHitsToDNN.value)
            ok = EnergyRecoBoostStore.Has("lambda_vec")
            print("EnergyRecoBoostStore has entry lambda_vec: ",ok)
            print("type of lambda_vec entry is :",EnergyRecoBoostStore.Type("lambda_vec"))
            print("get lambda_vec from EnergyRecoBoostStore")
            lambda_vector=std.vector['double'](range(MaxTotalHitsToDNN.value))
            # get the contents of the Energy Reco Store
            ok=EnergyRecoBoostStore.Get("lambda_vec", lambda_vector)
            print("Got lambda_vec from store",ok)
            print(lambda_vector.at(0))
            ok = EnergyRecoBoostStore.Has("digit_ts_vec")
            print("EnergyRecoBoostStore has entry digit_ts_vec: ",ok)
            print("type of digit_ts_vec entry is :",EnergyRecoBoostStore.Type("digit_ts_vec"))
            print("get digit_ts_vec from EnergyRecoBoostStore")
            digit_ts_vector=std.vector['double'](range(MaxTotalHitsToDNN.value))
            # get the contents of the Energy Reco Store
            ok=EnergyRecoBoostStore.Get("digit_ts_vec", digit_ts_vector)
            print("Got digit_ts_vec from store",ok)
            print(digit_ts_vector.at(0))
            ok = EnergyRecoBoostStore.Has("lambda_max")
            print("EnergyRecoBoostStore has entry lambda_max: ",ok)
            print("type of lambda_max entry is :",EnergyRecoBoostStore.Type("lambda_max"))
            print("get lambda_max from EnergyRecoBoostStore")
            lambda_max=ctypes.c_double()
            # get the contents of the Energy Reco Store
            EnergyRecoBoostStore.Get("lambda_max",lambda_max)
            print("Lambda_max is: ", lambda_max.value)
            ok = EnergyRecoBoostStore.Has("num_pmt_hits")
            print("EnergyRecoBoostStore has entry num_pmt_hits: ",ok)
            print("type of num_pmt_hits entry is :",EnergyRecoBoostStore.Type("num_pmt_hits"))
            print("get num_pmt_hits from EnergyRecoBoostStore")
            num_pmt_hits=ctypes.c_int()
            # get the contents of the Energy Reco Store
            EnergyRecoBoostStore.Get("num_pmt_hits",num_pmt_hits)
            print("Number of pmt hits is: ", num_pmt_hits.value)
            ok = EnergyRecoBoostStore.Has("num_lappd_hits")
            print("EnergyRecoBoostStore has entry num_lappd_hits: ",ok)
            print("type of num_lappd_hits entry is :",EnergyRecoBoostStore.Type("num_lappd_hits"))
            print("get num_lappd_hits from EnergyRecoBoostStore")
            num_lappd_hits=ctypes.c_int()
	    # get the contents of the Energy Reco Store
            EnergyRecoBoostStore.Get("num_lappd_hits",num_lappd_hits)
            print("Number of lappd hits is: ", num_lappd_hits.value)
            ok = EnergyRecoBoostStore.Has("TrueTrackLengthInWater")
            print("EnergyRecoBoostStore has entry TrueTrackLengthInWater: ",ok)
            print("type of TrueTrackLengthInWater entry is :",EnergyRecoBoostStore.Type("TrueTrackLengthInWater"))
            print("get TrueTrackLengthInWater from EnergyRecoBoostStore")
            TrueTrackLengthInWater=ctypes.c_float()
	    # get the contents of the Energy Reco Store
            EnergyRecoBoostStore.Get("TrueTrackLengthInWater",TrueTrackLengthInWater)
            print("TrueTrackLengthInWater is: ", TrueTrackLengthInWater.value)
            if(not ok):
                #if there is no entry in the Store the event did not pass the cuts so we don't reconstruct its track length in the water
                return 1
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
            features=numpy.array(features_list)
            features=features.reshape(1,-1)#reshape for model
            labels=numpy.array([TrueTrackLengthInWater.value])
        else:
            print("opening input data file ", self.inputdatafilename)
            filein = open(str(self.inputdatafilename))
            print("evts for predictions in: ", filein)
            # read into a pandas structure
            print("reading file with pandas")
            fileindata = pandas.read_csv(filein, sep=',', header=0)
            print(fileindata.head())
            print("closing file")
            filein.close()
            # convert to 2D numpy array                                                                                                                                                       
            print("converting to numpy array")
            Dataset = numpy.array(fileindata)
            # split the numpy array up into sub-arrays                                                                                                                                  
            features, lambdamax, labels, rest = numpy.split(Dataset,[2203,2204,2205],axis=1)
            # print info                                                                                                                                                                              
            print( "lambdamax ", lambdamax[:2], labels[:2])
            #print(features[0])                                                                                                                                        
            num_events, num_pixels = features.shape
            print(num_events, num_pixels)
        # Preprocess data and load model                                                                                                                                                                               
        #-----------------------------                                                                                                                                                                                 

        # rename variables for obfuscation
        test_x = features
        test_y = labels
        print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

        # Scale data to 0 mean and unit standard deviation(not when using the Store since you have a single sample).
        if(not self.fUseStore):                                                      
         print("scaling to 0 mean and unit std-dev")
         scaler = preprocessing.StandardScaler()
         x_transformed = scaler.fit_transform(test_x)  # are we ok doing fit_transform on test data?
         # scale the features                                                                                                                                                                                           
         features_transformed = scaler.transform(features)#NEED TO CHECK THIS WHEN RUNNING THE TRAIN_TEST TOOLCHAIN
        else:
         x_transformed=test_x
        # define keras model, loading weight from weights file                                                                                                                                                         
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
        y_predicted = model.predict(x_transformed)
        print(y_predicted.shape)

        # estimate accuracy on dataset using loaded weights                                                                                                                                                            
        print("evalulating model on test")
        scores = model.evaluate(test_x, test_y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        # Score with sklearn.
        print("scoring sk mse")
        score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
        print('MSE (sklearn): {0:f}'.format(score_sklearn))
        
        # Write to the output csv file or save the reconstructed track length in the Energy Reco BoostStore                                                                                                                                                                                
        #------------------------------
        if(self.fUseStore):
            DNNRecoLength=float(y_predicted[0,0])
            EnergyRecoBoostStore.Set("DNNRecoLength",DNNRecoLength)
            print("Storing the predicted track length in the water tank by the DNN in the Energy Reco BoostStore")
        else:
            if (self.predictionsfilename is None) or ( self.predictionsfilename == ''):
             # no output files today                                                                                                                                                                                    
             print("no output file specified, not writing to file")
             return 1
        
            print("writing predictions to output file "+self.predictionsfilename)
            # build a dataframe from the true and predicted track lengths                                                                                                                                                  
            print("building output dataframe")
            outputdataarray = numpy.concatenate((test_y, y_predicted),axis=1)
            outputdataframe=pandas.DataFrame(outputdataarray, columns=['TrueTrackLengthInWater','DNNRecoLength'])

            # append as additional columns to the input dataframe                                                                                                                                                          
            print("inserting True and Predicted lengths into file data")
            testfiledata.insert(2217, 'TrueTrackLengthInWater', outputdataframe['TrueTrackLengthInWater'].values, allow_duplicates="True")
            testfiledata.insert(2218, 'DNNRecoLength', outputdataframe['DNNRecoLength'].values, allow_duplicates="True")

            # write to csv file                                                                                                                                                                                            
            print("writing all data to "+self.predictionsfilename)
            testfiledata.to_csv(str(self.predictionsfilename), float_format = '%.3f')

            print("clearing session")
        K.clear_session()

        print("done; returning")
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
