/*
 * check.cpp
 *
 *  Created on: 23 Mar 2017
 *      Author: jkiesele
 */




#include <iostream>
#include "friendTreeInjector.h"
#include "TCanvas.h"

int main(){

	friendTreeInjector in;
	in.addFromFile("/afs/cern.ch/user/j/jkiesele/work/DeepLearning/DeepJet/Train/outfile.txt");

	in.showList();

	in.createChain();



	std::cout << in.getChain()->GetEntries() <<std::endl;

	TCanvas cv;
	in.getChain()->Draw("jet_pt","prob_isB>0.8");
	cv.Print("test.pdf");
}
