/*
 * makejetcuts.cpp
 *
 *  Created on: 4 Jul 2017
 *      Author: jkiesele
 */

#include "backToBack.h"


int main(int argc, char *argv[]){
	
	if(argc<3)return -1;
	TString infile=argv[1];
	TString outfile=argv[2];
	
	TFile * f=new TFile(infile,"READ");
    TDirectory * dir = (TDirectory*)f->Get("deepntuplizer");
    TTree* t;
    dir->GetObject("tree",t);

	backToBack bb(t);
	bb.outfile=outfile;
	bb.Loop();

	f->Close();
	delete f;
	
}
