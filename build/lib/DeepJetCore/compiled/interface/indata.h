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

#ifndef MAXBRANCHLENGTH
#define MAXBRANCHLENGTH 40000
#endif

namespace __hidden{
class indata{
public:
	static bool meanPadding;
	static bool doscaling;

	indata():max(0),offset_(0),mask_(-1){

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
	float getRaw(const size_t& b,const size_t& i) const {return buffer.at(b)[i];}
	float mean(const size_t& b) const {return means.at(b);}
	float std(const size_t& b) const {return norms.at(b);}
	float getDefault(const size_t& b);

	void allZero();

	void getEntry(size_t entry);

	void zeroAndGet(size_t entry) {
		allZero();
		getEntry(entry);
	}

	void setup(TTree* tree, const TString& treename="");

	size_t branchOffset(const size_t& i){
		return i*max;
	}

	size_t nfeatures() {return branches.size();}
	size_t nelements() {return buffer.size();}

	std::vector<float* > buffer;
	std::vector<std::vector<float>*> buffervec;

	std::vector<TBranch* > tbranches;

	size_t offset_;

	std::vector<float> norms,means;
	std::vector<TString> branches;
	int max;
    void setMask(int m){mask_=m;}
private:

    void handleReturns(int retcode, const TString& branchname)const;

    int mask_;
};



std::vector<__hidden::indata> createDataVector(std::vector< std::vector<TString>  >s_branches,
		std::vector< std::vector<double> > s_norms,
		std::vector< std::vector<double> > s_means,
		std::vector<int> s_max
);


}

#endif /* DEEPJET_MODULES_INTERFACE_INDATA_H_ */
