#include "PlotsTrackLengthAndEnergy.h"
#include "TCanvas.h"
#include "TMath.h"
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

    EnergyReco->Initialise("EnergyRecoStore.bs");

    unsigned long n_entries;
    auto get_ok = EnergyReco->Header->Get("TotalEntries", n_entries);
    std::cout<<"get total entries; "<<get_ok<<", n_entries: "<<n_entries<<std::endl;

    TCanvas *c1= new TCanvas();
    TCanvas *c2= new TCanvas();
    TCanvas *c3= new TCanvas();
    TCanvas *c4= new TCanvas();
    
    TH2D *lengthhist = new TH2D("True_RecoLength", "; MC Track Length [cm]; Reconstructed Track Length [cm]", 50, 0, 400., 50, 0., 400.);
    TH2D *energyhist = new TH2D("True_Reco_Energy", ";  E_{MC} [MeV]; E_{reco} [MeV]", 100, 0, 2000., 100, 0., 2000.);
    TH1D *lengthresol1 = new TH1D("wDNNRecolength", "ΔR = |L_{Reco}-L_{MC}| [cm]", 80, 0, 400);
    TH1D *lengthresol2 = new TH1D("wlambda_max", "ΔR = |lambda_{max}-L_{MC}| [cm]", 80, 0, 400);
    TH1D *energyresol1 = new TH1D("MC Energy", "MC Muon Energy [MeV]", 100, 0, 2000);
    TH1D *energyresol2 = new TH1D("BDT Energy", "Reco Muon Energy [MeV]", 100, 0, 2000);

    for(int i=0; i<n_entries; i++){
    double TrueTrackLengthInWater, DNNRecoLength, trueEnergy, BDTMuonEnergy, lambda_max;
    m_data->Stores.at("EnergyReco")->Get("TrueTrackLengthInWater",TrueTrackLengthInWater);
    m_data->Stores.at("EnergyReco")->Get("DNNRecoLength",DNNRecoLength);
    m_data->Stores.at("EnergyReco")->Get("trueE",trueEnergy);
    m_data->Stores.at("EnergyReco")->Get("BDTMuonEnergy",BDTMuonEnergy);
    m_data->Stores.at("EnergyReco")->Get("lambda_max",lambda_max);

    lengthhist->Fill(TrueTrackLengthInWater,DNNRecoLength);
    energyhist->Fill(trueEnergy,BDTMuonEnergy);
    lengthresol1->Fill(TMath::Abs(DNNRecoLength-TrueTrackLengthInWater));
    lengthresol2->Fill(TMath::Abs(lambda_max-TrueTrackLengthInWater));
    energyresol1->Fill(trueEnergy);
    energyresol2->Fill(BDTMuonEnergy);
    }
    c1->cd();
    lengthhist->SetStats(0);
    lengthhist->Draw("ColZ");
    c1->Draw();
    c1->SaveAs("MC_recolength.png");
    
    c2->cd();
    energyhist->SetStats(0);
    energyhist->Draw("ColZ");
    c2->Draw();
    c2->SaveAs("MC_recoE.png");
    
    c3->cd();
    energyresol1->Draw();
    energyresol2->Draw("Same");
    c3->SaveAs("resol_energy.png");
    
    c4->cd();
    lengthresol1->Draw();
    lengthresol2->Draw("Same");
    c4->SaveAs("resol_length.png");
    
  return true;
}


bool PlotsTrackLengthAndEnergy::Finalise(){
  
  return true;
}
