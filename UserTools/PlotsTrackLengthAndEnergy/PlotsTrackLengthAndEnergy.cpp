#include "PlotsTrackLengthAndEnergy.h"
#include "TCanvas.h"
#include "TH2F.h"
PlotsTrackLengthAndEnergy::PlotsTrackLengthAndEnergy():Tool(){}


bool PlotsTrackLengthAndEnergy::Initialise(std::string configfile, DataModel &data){

  /////////////////// Useful header ///////////////////////
  if(configfile!="") m_variables.Initialise(configfile); // loading config file
  //m_variables.Print();

  m_data= &data; //assigning transient data pointer
  /////////////////////////////////////////////////////////////////
  
  return true;
}


bool PlotsTrackLengthAndEnergy::Execute(){

    BoostStore* EnergyReco = new BoostStore(true,2);

    EnergyReco->Initialise("EnergyRecoFINAL.bs");

    unsigned long n_entries;
    auto get_ok = EnergyReco->Header->Get("TotalEntries", n_entries);
    std::cout<<"get total entries; "<<get_ok<<", n_entries: "<<n_entries<<std::endl;

    TCanvas *c1= new TCanvas();
    c1->cd();
    TH2F *lengthhist = new TH2F("True_RecoLength", "; MC Track Length [cm]; Reconstructed Track Length [cm]", 50, 0, 400., 50, 0., 400.);
    TH2F *energyhist = new TH2F("True_Reco_Energy", ";  E_{MC} [MeV]; E_{reco} [MeV]", 100, 0, 2000., 100, 0., 2000.);

    for(int i=0; i<n_entries; i++){
    double TrueTrackLengthInWater, DNNRecoLength, trueEnergy, BDTMuonEnergy;
    m_data->Stores.at("EnergyReco")->Get("TrueTrackLengthInWater",TrueTrackLengthInWater);
    m_data->Stores.at("EnergyReco")->Get("DNNRecoLength",DNNRecoLength);
    m_data->Stores.at("EnergyReco")->Get("trueE",trueEnergy);
    m_data->Stores.at("EnergyReco")->Get("BDTMuonEnergy",BDTMuonEnergy);

    lengthhist->Fill(TrueTrackLengthInWater,DNNRecoLength);
    energyhist->Fill(trueEnergy,BDTMuonEnergy);
    }
    lengthhist->SetStats(0);
    lengthhist->Draw("ColZ");
    c1->Update();
    c1->SaveAs("MC_recolength.png");
    
    energyhist->SetStats(0);
    energyhist->Draw("ColZ");
    c1->Update();
    c1->SaveAs("MC_recoE.png");
    

  return true;
}


bool PlotsTrackLengthAndEnergy::Finalise(){
  
  return true;
}
