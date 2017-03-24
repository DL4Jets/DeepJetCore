/*
 * friendTreeInjector.h
 *
 *  Created on: 23 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_ANALYSIS_INTERFACE_FRIENDTREEINJECTOR_H_
#define DEEPJET_ANALYSIS_INTERFACE_FRIENDTREEINJECTOR_H_

#include "TString.h"
#include "TChain.h"

class friendTreeInjector{
public:
	friendTreeInjector();
	~friendTreeInjector();

	void addFromFile(const TString& filename);

	void createChain();

	TChain* getChain(){return chain_;}

	void showList()const;

private:

	void resetChain();

	std::vector<std::vector<TString> > treesandfriends_;
	std::vector<TString> friendaliases_;

	TChain* chain_;
	std::vector<TChain*> friendchains_;
};



#endif /* DEEPJET_ANALYSIS_INTERFACE_FRIENDTREEINJECTOR_H_ */
