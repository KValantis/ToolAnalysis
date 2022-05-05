#include "MySecondTool.h"

MySecondTool::MySecondTool():Tool(){}


bool MySecondTool::Initialise(std::string configfile, DataModel &data){

  /////////////////// Useful header ///////////////////////
  if(configfile!="") m_variables.Initialise(configfile); // loading config file
  //m_variables.Print();

  m_data= &data; //assigning transient data pointer
  /////////////////////////////////////////////////////////////////

  return true;
}


bool MySecondTool::Execute(){

  return true;
}


bool MySecondTool::Finalise(){

  return true;
}
