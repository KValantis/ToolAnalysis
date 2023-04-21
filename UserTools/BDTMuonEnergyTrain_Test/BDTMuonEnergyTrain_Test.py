# BDTMuonEnergyTrain_Test Tool script
# ------------------
import sys
import numpy as np
import pandas as pd
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
from Tool import *

class BDTMuonEnergyTrain_Test(Tool):
    
    # declare member variables here
    E_threshold=ctypes.c_double()
    inputdatafilename=std.string("")
    weightsfilename=std.string("")
    
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("BDT_NuE_threshold", self.E_threshold)
        self.m_variables.Get("BDTMuonEnergyTrainingInputDataFile", self.inputdatafilename)
        self.m_variables.Get("BDTMuonEnergyWeightsFile", self.weightsfilename)
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

thistool = BDTMuonEnergyTrain_Test()

def SetToolChainVars(m_data_in, m_variables_in, m_log_in):
    return thistool.SetToolChainVars(m_data_in, m_variables_in, m_log_in)

def Initialise():
    return thistool.Initialise()

def Execute():
    # Set random seed to improve reproducibility
    seed = 170
    np.random.seed(seed)
    
    E_low=0
    E_high=1000*thistool.E_threshold #convert to MeV
    div=100
    bins = int((E_high-E_low)/div)
print('bins: ', bins)

print( "--- opening file with input variables!") 
#--- events for training ---
filein = open(str(thistool.inputdatafilename))
print("evts for training in: ",filein)
df00=pd.read_csv(filein)
df0=df00[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','recoTrackLengthInMRD','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
#dfsel=df0.loc[df0['neutrinoE'] < E_threshold]
dfsel=df0.dropna()
print("df0.head(): ", df0.head())

#print to check:
print("check training sample: \n",dfsel.head())
#check fr NaN values:
print("The dimensions of training sample ",dfsel.shape)
assert(dfsel.isnull().any().any()==False)

#--- normalisation-training sample:
#dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['TrueTrackLengthInMrd']/200., dfsel['diffDirAbs'], dfsel['recoDWallR']/152.4, dfsel['recoDWallZ']/198., dfsel['totalLAPPDs']/1000., dfsel['totalPMTs']/1000., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['recoTrackLengthInMRD']/200., dfsel['diffDirAbs'], dfsel['recoDWallR'], dfsel['recoDWallZ'], dfsel['totalLAPPDs']/200., dfsel['totalPMTs']/200., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
print("check normalisation: ", dfsel_n.head())
#discard events with no reconstructed mrd tracks
MRDTrackLength=dfsel_n['recoTrackLengthInMRD']
i=0
a=[]
for y in MRDTrackLength:
   if y<0:
     print("MRDTrackLength:",y,"Event:",i)
     a.append(i)
   i=i+1
dfsel_n1=dfsel_n.drop(dfsel_n.index[a])
dfsel1=dfsel.drop(dfsel.index[a])
#--- prepare training & test sample for BDT:
arr_hi_E0 = np.array(dfsel_n1[['DNNRecoLength','recoTrackLengthInMRD','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']])
#arr_hi_E1 = np.delete(arr_hi_E0, 1, axis=1)
arr3_hi_E0 = np.array(dfsel1[['trueKE']])
 
#---- random split of events ----
rnd_indices = np.random.rand(len(arr_hi_E0)) < 1. #< 0.50
print(rnd_indices[0:5])
#--- select events for training/test:
arr_hi_E0B = arr_hi_E0[rnd_indices]
print(arr_hi_E0B[0:5])
arr2_hi_E_n = arr_hi_E0B #.reshape(arr_hi_E0B.shape + (-1,))
arr3_hi_E = arr3_hi_E0[rnd_indices]
##--- select events for prediction: -- in future we need to replace this with data sample!
#evts_to_predict = arr_hi_E0[~rnd_indices]
#evts_to_predict_n = evts_to_predict #.reshape(evts_to_predict.shape + (-1,))
#test_data_trueKE_hi_E = arr3_hi_E0[~rnd_indices]

#printing..
print('events for training: ',len(arr3_hi_E)) #,' events for predicting: ',len(test_data_trueKE_hi_E)) 
print('initial train shape: ',arr3_hi_E.shape) #," predict: ",test_data_trueKE_hi_E.shape)

########### BDTG ############
n_estimators=500
params = {'n_estimators':n_estimators, 'max_depth': 50,
          'learning_rate': 0.025, 'loss': 'absolute_error'} 

print("arr2_hi_E_n.shape: ",arr2_hi_E_n.shape)
#--- select 70% of sample for training and 30% for testing:
offset = int(arr2_hi_E_n.shape[0] * 0.7) 
arr2_hi_E_train, arr3_hi_E_train = arr2_hi_E_n[:offset], arr3_hi_E[:offset].reshape(-1)  # train sample
arr2_hi_E_test, arr3_hi_E_test   = arr2_hi_E_n[offset:], arr3_hi_E[offset:].reshape(-1)  # test sample
 
print("train shape: ", arr2_hi_E_train.shape," label: ",arr3_hi_E_train.shape)
print("test shape: ", arr2_hi_E_test.shape," label: ",arr3_hi_E_test.shape)
    
print("training BDTG...")
net_hi_E = ensemble.GradientBoostingRegressor(**params)
model = net_hi_E.fit(arr2_hi_E_train, arr3_hi_E_train)
net_hi_E

# save the model to disk
filename = str(thistool.weightsfilename)
pickle.dump(model, open(filename, 'wb'))

mse = mean_squared_error(arr3_hi_E_test, net_hi_E.predict(arr2_hi_E_test)) 
print("MSE: %.4f" % mse)
print("events at training & test samples: ", len(arr_hi_E0))
print("events at train sample: ", len(arr2_hi_E_train))
print("events at test sample: ", len(arr2_hi_E_test))
 
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
 
for i, y_pred in enumerate(net_hi_E.staged_predict(arr2_hi_E_test)):
    test_score[i] = net_hi_E.loss_(arr3_hi_E_test, y_pred)

fig,ax=plt.subplots(ncols=1, sharey=True)
ax.plot(np.arange(params['n_estimators']) + 1, net_hi_E.train_score_, 'b-',
             label='Training Set Error')
ax.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Error')
ax.set_ylim(0.,500.)
ax.set_xlim(0.,n_estimators)
ax.legend(loc='upper right')
ax.set_ylabel('Absolute Errors [MeV]')
ax.set_xlabel('Number of Estimators')
ax.yaxis.set_label_coords(-0.1, 0.6)
ax.xaxis.set_label_coords(0.85, -0.08)
plt.savefig("error_train_test.png")
    return thistool.Execute()

def Finalise():
    return thistool.Finalise()
