#include "MyFirstTool.h"
#include <iostream>
using namespace std;
MyFirstTool::MyFirstTool():Tool(){}


bool MyFirstTool::Initialise(std::string configfile, DataModel &data){

  /////////////////// Useful header ///////////////////////
  if(configfile!="") m_variables.Initialise(configfile); // loading config file
  //m_variables.Print();

  m_data= &data; //assigning transient data pointer
  /////////////////////////////////////////////////////////////////

  return true;
}


bool MyFirstTool::Execute(){
  cout<<"hello World, this is my first tool";
  return true;
}


bool MyFirstTool::Finalise(){

  return true;
}
