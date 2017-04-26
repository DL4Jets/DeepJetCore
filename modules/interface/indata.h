/*
 * indata.h
 *
 *  Created on: 8 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_INDATA_H_
#define DEEPJET_MODULES_INTERFACE_INDATA_H_


#include <iostream>
#include <stdint.h>
#include "TString.h"
#include <string>
#include <vector>
#include "TFile.h"
#include "TTree.h"

#define MAXBRANCHLENGTH 200

namespace __hidden{
class indata{
public:
	indata():max(0),offset_(0){

	}

	void createFrom(std::vector<TString>  s_branches,
			 std::vector<double>  s_norms,
			 std::vector<double>  s_means,
			int s_max);

	void setSize(size_t i);

	~indata();

	size_t getMax()const{
		return max;
	}

	float getData(const size_t& b,const size_t& i);
	float getDefault(const size_t& b);

	void allZero();

	void getEntry(size_t entry);

	void zeroAndGet(size_t entry) {
		allZero(); 
		getEntry(entry);
	}

	void setup(TTree* tree);

	size_t branchOffset(const size_t& i){
		return i*max;
	}

	size_t nfeatures() {return branches.size();}
	size_t nelements() {return buffer.size();}

	std::vector<float* > buffer;

	std::vector<TBranch* > tbranches;

	size_t offset_;

	std::vector<float> norms,means;
	std::vector<TString> branches;
	int max;
};



std::vector<__hidden::indata> createDataVector(std::vector< std::vector<TString>  >s_branches,
		std::vector< std::vector<double> > s_norms,
		std::vector< std::vector<double> > s_means,
		std::vector<int> s_max
);


}

#endif /* DEEPJET_MODULES_INTERFACE_INDATA_H_ */
