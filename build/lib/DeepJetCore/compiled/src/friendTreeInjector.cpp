/*
 * friendTreeInjector.cpp
 *
 *  Created on: 23 Mar 2017
 *      Author: jkiesele
 */

#include "friendTreeInjector.h"
#include <fstream>
#include <iostream>


friendTreeInjector::friendTreeInjector(TString sourcetreename):
        chain_(0),
        sourcetree_('/'+sourcetreename){}
friendTreeInjector::~friendTreeInjector(){
	resetChain();
}

void friendTreeInjector::addFromFile(const TString& filename, const TString& alias){

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
		toinject.push_back(b);
	}
	friendaliases_.push_back(alias);
	std::cout << "added alias "<<alias <<std::endl;

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
		throw std::runtime_error("friendTreeInjector::addFromFile: file lists not same length");
	}

	//else
	for( auto& o:treesandfriends_){
		const TString& orig=o.at(0);
		for(size_t i=0;i<originroots.size();i++){
			if(originroots.at(i) == orig){
				o.push_back(toinject.at(i));
				//std::cout << toinject.at(i) << std::endl;
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
	friendchains_ = std::vector<TChain*> (treesandfriends_.at(0).size()-1,0);
	for(size_t i=0;i<treesandfriends_.at(0).size()-1;i++){
	    TString s="";
	    s+=i;
	    friendchains_.at(i)=new TChain(s,s);
	}
	for(size_t i=0;i<treesandfriends_.size();i++){
	    TString basetree=treesandfriends_.at(i).at(0)+sourcetree_;
	    //std::cout << basetree << std::endl;
		chain_->AddFile(basetree);
		for(size_t j=1;j<treesandfriends_.at(i).size();j++){
		    TString friendtree=treesandfriends_.at(i).at(j)+"/tree";
		    //std::cout << j-1<<' '<<friendtree << std::endl;
		    if(friendtree!="DUMMY/tree")
		        friendchains_.at(j-1)->AddFile(friendtree);
		}
	}
	for(size_t i=0;i<friendchains_.size();i++){
		size_t entries=chain_->GetEntries();
		size_t friendentries=friendchains_.at(i)->GetEntries();
		//std::cout << entries << ' '<< friendentries << std::endl;
		if(friendentries!=0 && entries!=friendentries)
			throw std::out_of_range("friendTreeInjector::createChain: trees don't have same number of entries.\nIs is possible that the test data was not converted using --testdatafor?");
		if(friendentries)
		    chain_->AddFriend(friendchains_.at(i),friendaliases_.at(i));
	}

}

void friendTreeInjector::resetChain(){

}

