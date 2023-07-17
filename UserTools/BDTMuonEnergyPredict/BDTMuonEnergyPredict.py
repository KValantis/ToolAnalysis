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
import ROOT

class BDTMuonEnergyPredict(Tool):
    
    # declare member variables here
    weightsfilename = std.string("")
    
    def Initialise(self):
        self.m_log.Log(__file__+" Initialising", self.v_debug, self.m_verbosity)
        self.m_variables.Print()
        self.m_variables.Get("BDTMuonModelFile", self.weightsfilename)
        header=['TrueTrackLengthInWater', 'DNNRecoLength', 'TrueMuonEnergy', 'BDTMuonEnergy']
        f = open('TrackLengthAndEnergy.csv', 'w', encoding='UTF8', newline='')
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        f.close()

        return 1
    
    def Execute(self):
        self.m_log.Log(__file__+" Executing", self.v_debug, self.m_verbosity)
        #Load Data
        EnergyRecoBoostStore=cppyy.gbl.BoostStore(True, 2)#define the energy boost store class object to load the variables for the BDT training
        EnergyRecoBoostStore = self.m_data.Stores.at("EnergyReco")
        #Retrieve the required variables from the Store
        EnergyRecoBoostStore.Print(False)
        get_ok = EnergyRecoBoostStore.Has("num_pmt_hits")
        num_pmt_hits=ctypes.c_int(0)
        if not get_ok:
            print("There is no entry in Energy Reco boost store.")
            return 1
        if get_ok:
            print("EnergyRecoBoostStore has entry num_pmt_hits: ", get_ok)
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
        RecoEventStore = cppyy.gbl.BoostStore(True, 0)
        RecoEventStore = self.m_data.Stores.at("RecoEvent")
        ok = RecoEventStore.Has("DNNRecoLength")
        DNNRecoLength=ctypes.c_double(0)
        if ok:
            print("RecoEventStore has entry DNNRecoLength: ",ok)
            print("type of DNNRecoLength entry is :",RecoEventStore.Type("DNNRecoLength"))
            print("Getting DNNRecoLength from RecoEventStore")
            RecoEventStore.Get("DNNRecoLength",DNNRecoLength)
        print("The reconstructed track length in the water by the DNN is: ", DNNRecoLength.value)
        ok = EnergyRecoBoostStore.Has("recoTrackLengthInMrd")
        recoTrackLengthInMrd=ctypes.c_double(0)
        if ok:
            print("EnergyRecoBoostStore has entry recoTrackLengthInMrd: ",ok)
            print("type of recoTrackLengthInMrd entry is :",EnergyRecoBoostStore.Type("recoTrackLengthInMrd"))
            print("Getting recoTrackLengthInMrd from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("recoTrackLengthInMrd",recoTrackLengthInMrd)
        print("The reconstructed track length in the MRD is: ", recoTrackLengthInMrd.value)
        ok = EnergyRecoBoostStore.Has("diffDirAbs2")
        diffDirAbs=ctypes.c_float(0)
        if ok:
            print("EnergyRecoBoostStore has entry diffDirAbs: ",ok)
            print("type of diffDirAbs entry is :",EnergyRecoBoostStore.Type("diffDirAbs2"))
            print("Getting diffDirAbs from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("diffDirAbs2",diffDirAbs)
        print("DiffDirAbs is: ", diffDirAbs.value)
        ok = EnergyRecoBoostStore.Has("recoDWallR2")
        recoDWallR=ctypes.c_float(0)
        if ok:
            print("EnergyRecoBoostStore has entry recoDWallR: ",ok)
            print("type of recoDWallR entry is :",EnergyRecoBoostStore.Type("recoDWallR2"))
            print("Getting recoDWallR from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("recoDWallR2",recoDWallR)
        print("RecoDWallR is: ", recoDWallR.value)
        ok = EnergyRecoBoostStore.Has("recoDWallZ2")
        recoDWallZ=ctypes.c_float(0)
        if(ok):
            print("EnergyRecoBoostStore has entry recoDWallZ: ",ok)
            print("type of recoDWallZ entry is :",EnergyRecoBoostStore.Type("recoDWallZ2"))
            print("Getting recoDWallZ from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("recoDWallZ2",recoDWallZ)
        print("RecoDWallZ is: ", recoDWallZ.value)
        ok = EnergyRecoBoostStore.Has("vtxVec")
        vtx_position=cppyy.gbl.Position()
        if ok:
            print("EnergyRecoBoostStore has entry vtxVec: ",ok)
            print("type of vtxVec entry is :", EnergyRecoBoostStore.Type("vtxVec"))
            print("Getting vtxVec from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("vtxVec", vtx_position)
        vtxX=vtx_position.X()
        print("VtxX is: ", vtxX)
        vtxY=vtx_position.Y()
        print("VtxY is: ", vtxY)
        vtxZ=vtx_position.Z()
        print("VtxZ is: ", vtxZ)
        ok = EnergyRecoBoostStore.Has("trueE")
        trueE=ctypes.c_double(0)
        if ok:
            print("EnergyRecoBoostStore has entry trueE: ",ok)
            print("type of trueE entry is :",EnergyRecoBoostStore.Type("trueE"))
            print("Getting trueE from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("trueE",trueE)
        print("The MC muon energy is: ", trueE.value)

        #Create features and labels and preprocess data for the model
        features_list=[]
        features_list.append(DNNRecoLength.value/600.)
        features_list.append(recoTrackLengthInMrd.value/200.)
        features_list.append(diffDirAbs.value)
        features_list.append(recoDWallR.value)
        features_list.append(recoDWallZ.value)
        features_list.append(num_lappd_hits.value/200.)
        features_list.append(num_pmt_hits.value/200.)
        features_list.append(vtxX/150.)
        features_list.append(vtxY/200.)
        features_list.append(vtxZ/150.)
        features=np.array(features_list).reshape(1,10)
        labels=np.array([trueE.value])

        # load the model from disk
        print("loading model")
        loaded_model = pickle.load(open(str(self.weightsfilename), 'rb'))

        #predicting...
        print("predicting")
        recoEnergy = loaded_model.predict(features)
        print("True:",trueE.value,"Reco:",recoEnergy[0])

        #Set the BDTMuonEnergy in the EnergyReco boost store to be loaded by other tools
        BDTMuonEnergy=ctypes.c_double(recoEnergy[0])
        RecoEventStore.Set("BDTMuonEnergy", BDTMuonEnergy)

        ok = EnergyRecoBoostStore.Has("TrueTrackLengthInWater")
        TrueTrackLengthInWater=ctypes.c_float(0)
        if ok:
            print("EnergyRecoBoostStore has entry TrueTrackLengthInWater: ",ok)
            print("type of TrueTrackLengthInWater entry is :",EnergyRecoBoostStore.Type("TrueTrackLengthInWater"))
            print("Getting TrueTrackLengthInWater from EnergyRecoBoostStore")
            EnergyRecoBoostStore.Get("TrueTrackLengthInWater",TrueTrackLengthInWater)
        print("TrueTrackLengthInWater is: ", TrueTrackLengthInWater.value)
        data=[TrueTrackLengthInWater.value,DNNRecoLength.value,trueE.value,BDTMuonEnergy.value]
        f = open('TrackLengthAndEnergy.csv', 'a', encoding='UTF8', newline='')
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()

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
    return thistool.Execute()

def Finalise():
    return thistool.Finalise()
