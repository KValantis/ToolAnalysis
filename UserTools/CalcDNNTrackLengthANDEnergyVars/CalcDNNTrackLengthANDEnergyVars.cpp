#include "CalcDNNTrackLengthANDEnergyVars.h"

CalcDNNTrackLengthANDEnergyVars::CalcDNNTrackLengthANDEnergyVars():Tool(){}


bool CalcDNNTrackLengthANDEnergyVars::Initialise(std::string configfile, DataModel &data){

  /////////////////// Useful header ///////////////////////
  if(configfile!="") m_variables.Initialise(configfile); // loading config file
  //m_variables.Print();

  m_data= &data; //assigning transient data pointer
  /////////////////////////////////////////////////////////////////

  return true;
}


bool CalcDNNTrackLengthANDEnergyVars::Execute(){

  return true;
}


bool CalcDNNTrackLengthANDEnergyVars::Finalise(){

  return true;
}
