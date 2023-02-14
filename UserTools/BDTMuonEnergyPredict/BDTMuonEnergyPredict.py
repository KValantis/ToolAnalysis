# python Tool script
# ------------------
from Tool import *
import sys
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
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
import pickle

class BDTMuonEnergyPredict(Tool):
    
    # declare member variables here
    E_threshold=ctypes.c_double()
    inputdatafilename=std.string("")
    predictionsdatafilename=std.string("")
    modelfilename=std.string("")
    
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("BDT_NuE_threshold", self.E_threshold)
        self.m_variables.Get("MuonEnergyInputDataFile", self.inputdatafilename)
        self.m_variables.Get("MuonEnergyPredictionsFile", self.predictionsdatafilename)
        self.m_variables.Get("BDTMuonModelFile", self.modelfilename)
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

thistool = BDTMuonEnergyPredict()

def SetToolChainVars(m_data_in, m_variables_in, m_log_in):
    return thistool.SetToolChainVars(m_data_in, m_variables_in, m_log_in)

def Initialise():
    return thistool.Initialise()

def Execute():
# Set TF random seed to improve reproducibility
    seed = 170
    np.random.seed(seed)
    E_low=0
    E_high=2000
    div=100
    bins = int((E_high-E_low)/div)
    print('bins: ', bins)

    print( "--- opening file with input variables"+str(thistool.inputdatafilename)) 
    #--- events for prediction ---
    filein = open(str(thistool.inputdatafilename))
    print("evts for prediction in: ",filein)
    df00=pd.read_csv(filein)
    df0=df00[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater',#'neutrinoE',
    'trueKE','diffDirAbs','recoTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
    #dfsel=df0.loc[df0['neutrinoE'] < E_threshold]
    dfsel=df0
    
    #print to check:
    print("check prediction sample: ",dfsel.head())
#    print(dfsel.iloc[5:10,0:5])
    #check fr NaN values:
    assert(dfsel.isnull().any().any()==False)

    #--- normalisation-prediction sample:
    dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['recoTrackLengthInMrd']/200., dfsel['diffDirAbs'], dfsel['recoDWallR'], dfsel['recoDWallZ'], dfsel['totalLAPPDs']/200., dfsel['totalPMTs']/200., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
    print("chehck normalisation: ", dfsel_n.head())

    #--- prepare prediction sample for BDT:
    #discard events with no reconstructed mrd tracks
    MRDTrackLength=dfsel_n['recoTrackLengthInMrd']
    i=0
    a=[]
    for y in MRDTrackLength:
         if y<0:
            print("MRDTrackLength:",y,"Event:",i)
            a.append(i)
         i=i+1
    dfsel_n1=dfsel_n.drop(dfsel_n.index[a])
    dfsel1=dfsel.drop(dfsel.index[a]) 
    arr_hi_E0 = np.array(dfsel_n1[['DNNRecoLength','recoTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']])
    arr3_hi_E0 = np.array(dfsel1[['trueKE']])
 
    #printing..
    print(' events for predicting: ',len(arr3_hi_E0)) 

    ########### BDTG ############
    n_estimators=1000

 
    # load the model from disk
    loaded_model = pickle.load(open(str(thistool.modelfilename), 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result) 

    #predicting...
    print("events for energy reco: ", len(arr3_hi_E0)) 
    BDTGoutput_E = loaded_model.predict(arr_hi_E0)

    Y=[0 for j in range (0,len(arr3_hi_E0))]
    for i in range(len(arr3_hi_E0)):
        Y[i] = 100.*(arr3_hi_E0[i]-BDTGoutput_E[i])/(1.*arr3_hi_E0[i])
#        print("MC Energy: ", test_data_trueKE_hi_E[i]," Reco Energy: ",BDTGoutput_E[i]," DE/E[%]: ",Y[i])

    df1 = pd.DataFrame(arr3_hi_E0,columns=['MuonEnergy'])
    df2 = pd.DataFrame(BDTGoutput_E,columns=['RecoE'])
    df_final = pd.concat([df1,df2],axis=1)
 
    #-logical tests:
    print("checking..."," df0.shape[0]: ",df1.shape[0]," len(y_predicted): ", len(BDTGoutput_E)) 
    assert(df1.shape[0]==len(BDTGoutput_E))
    assert(df_final.shape[0]==df2.shape[0])

    #save results to .csv:  
    if thistool.predictionsdatafilename == 'NA':
        return 1
    df_final.to_csv(str(thistool.predictionsdatafilename), float_format = '%.3f')
    return thistool.Execute()

def Finalise():
    return thistool.Finalise()
