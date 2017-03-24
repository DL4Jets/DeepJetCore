/*
 * friendTreeInjector.cpp
 *
 *  Created on: 23 Mar 2017
 *      Author: jkiesele
 */

#include "friendTreeInjector.h"
#include <fstream>
#include <iostream>

friendTreeInjector::friendTreeInjector():chain_(0){}
friendTreeInjector::~friendTreeInjector(){
	resetChain();
}

void friendTreeInjector::addFromFile(const TString& filename){

	//add to treesandfriends_

	std::vector<TString> originroots,toinject;

	std::ifstream file(filename.Data(), std::ifstream::in);
	if(!file){
		std::cerr << "friendTreeInjector::addFromFile: could not open file "<< filename <<std::endl;
	}
	TString a,b;
	while (file >> a >> b)
	{
		originroots.push_back(a);
		std::cout << a << ' ' << b <<std::endl;
		toinject.push_back(b);
	}

	if(treesandfriends_.size()<1){
		for(size_t i=0;i<originroots.size();i++){
			std::vector<TString> orig(1, originroots.at(i));
			orig.push_back(toinject.at(i));
			treesandfriends_.push_back(orig);
		}
		return;
	}
	//check for size
	if(originroots.size()!=treesandfriends_.size()){
		//throw something
	}

	//else
	for( auto& o:treesandfriends_){
		const TString& orig=o.at(0);
		for(size_t i=0;i<originroots.size();i++){
			if(originroots.at(i) == orig){
				o.push_back(toinject.at(i));
				break;
			}
		}
	}


}

void friendTreeInjector::showList()const{
	for(const auto& s:treesandfriends_){
		for(const auto& b:s){
			std::cout << b <<' ';
		}
		std::cout << std::endl;
	}
}

void friendTreeInjector::createChain(){
	resetChain();
	chain_ = new TChain();
	friendchains_ = std::vector<TChain*> (treesandfriends_.at(0).size()-1, new TChain());
	for(size_t i=0;i<treesandfriends_.size();i++){
		chain_->AddFile(treesandfriends_.at(i).at(0)+"/deepntuplizer/tree");
		for(size_t j=1;j<treesandfriends_.at(i).size();j++){
			friendchains_.at(j-1)->AddFile(treesandfriends_.at(i).at(j)+"/tree");
		}
	}
	for(size_t i=0;i<friendchains_.size();i++){
		chain_->AddFriend(friendchains_.at(i));
	}

}

void friendTreeInjector::resetChain(){

}

//std::vector<std::vector<TString> > treesandfriends_;
//std::vector<TString> friendaliases_;
//
//TChain* chain_;
