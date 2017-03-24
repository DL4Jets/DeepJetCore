/*
 * check.cpp
 *
 *  Created on: 23 Mar 2017
 *      Author: jkiesele
 */




#include <iostream>
#include "friendTreeInjector.h"
#include "rocCurveCollection.h"

int main(){

	//no GUI libs in the miniconda root installation!


	friendTreeInjector in;
	in.addFromFile("/afs/cern.ch/user/j/jkiesele/work/DeepLearning/DeepJet/Train/outfile.txt");

	//in.showList();

	in.createChain();

	std::cout << in.getChain()->GetEntries() <<std::endl;

	//simple Tree->Draw plotting


	rocCurveCollection rocColl;

	rocColl.addROC("B vs light deepCSV", "deepFlavourJetTags_probb","isB" ,"isUDS+isG","purple");
	rocColl.addROC("B vs light newTrain","prob_isB",                "isB" ,"isUDS+isG","BlAcK");
	rocColl.addROC("C vs light newTrain","prob_isC",                "isC" ,"isUDS+isG+isB","green");

	rocColl.printRocs(in.getChain(),"test3.pdf");

	//or use the chain as a normal TChain, loop, do plots etc.

}
