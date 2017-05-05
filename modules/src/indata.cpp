/*
 * indata.cpp
 *
 *  Created on: 8 Mar 2017
 *      Author: jkiesele
 */

#include "../interface/indata.h"


namespace __hidden{

void indata::setSize(size_t i){
	norms.resize(i,1);
	means.resize(i,0);
	branches.resize(i);
	tbranches.resize(i,0);
	buffer.resize(i,0);
}


void indata::createFrom(std::vector<TString>  s_branches,
		std::vector<double>  s_norms,
		std::vector<double>  s_means,
		int s_max){

	if(s_branches.size() != s_norms.size() || s_norms.size() != s_means.size())
		throw std::runtime_error("indata::createFrom inputs must have same size");

	size_t branchlength=s_branches.size();
	setSize(branchlength);

	max = s_max;
	for(int j=0;j<s_branches.size();j++){
		branches.at(j)=s_branches.at(j);
		norms.at(j)=s_norms.at(j);
		means.at(j)=s_means.at(j);
	}
}

indata::~indata(){
	//	for(auto& b:buffer)
	//		if(b) delete b;
	//	for(auto& b:branches)
	//		if(b) delete b;
}

float indata::getData(const size_t& b,const size_t& i){
	float ret= buffer.at(b)[i];
	ret -= means.at(b);
	ret /= norms.at(b);
	return ret;
}

float indata::getDefault(const size_t& b) {
	return (0 - means.at(b)) / norms.at(b);
}

void indata::allZero(){
	for(auto& c:buffer)
		for(int i=0;i<max;i++)
			c[i]=0;
}

void indata::getEntry(size_t entry){
	for(auto& b:tbranches)
		b->GetEntry(entry);
}

void indata::setup(TTree* tree){
	if(MAXBRANCHLENGTH<max)
		throw std::runtime_error("max larger than buffer! (clean up here needed: TBI)");

	for(auto& b: buffer)
		b=new float[MAXBRANCHLENGTH];

	for(size_t i=0;i<branches.size();i++){
		tbranches.at(i)=new TBranch();
		tree->SetBranchAddress(branches.at(i),buffer.at(i),&tbranches.at(i));
	}
}



std::vector<__hidden::indata> createDataVector(std::vector< std::vector<TString>  >s_branches,
		std::vector< std::vector<double> > s_norms,
		std::vector< std::vector<double> > s_means,
		std::vector<int> s_max){

	std::vector<__hidden::indata>  alldata;
	size_t offset=0;
	for(int i=0;i<s_branches.size();i++){

		__hidden::indata data_config;
		data_config.createFrom(s_branches.at(i),s_norms.at(i),s_means.at(i),s_max.at(i));

		data_config.offset_=offset;
		alldata.push_back(data_config);

		offset+=data_config.branches.size()*data_config.max;
	}


	return alldata;
}

}



