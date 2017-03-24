//#include "MyROCS.h"
#include <TString.h>
#include <iostream>

int myplots()
{
  
  TFile *f = new TFile("ntuple_ttbar_2016_test.root","read");
  TTree *treettbar =  (TTree*)f->Get("deepntuplizer/tree");
  MyROCS ttbar(treettbar);
  // CSV
  ttbar.Loop( TString("KERAS_result_val.root"), TString("KERAS_result_val.root"),true,TString("talk_CSV_ttbar"), 0,0 );


  return 0;
  
}
