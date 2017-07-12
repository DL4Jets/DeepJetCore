/*
 * rocCurve.h
 *
 *  Created on: 24 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_ANALYSIS_INTERFACE_ROCCURVE_H_
#define DEEPJET_ANALYSIS_INTERFACE_ROCCURVE_H_

#include <vector>
#include "TGraph.h"
#include "TH1D.h"
#include "TChain.h"
#include <iostream>

class rocCurve{
public:
	rocCurve();
	rocCurve(const TString& name);
	rocCurve(const TString& name, const TString& probability, const TString& truth, const TString& vetotruth, const TString& cuts="");
	rocCurve(const TString& name, const TString& probability, const TString& truth, const TString& vetotruth, int linecol, int linestyle,
	        const TString& cuts="",const TString& invalidateif="");

	~rocCurve();

	void addTruth(const TString& t){
		truths_.push_back(t);
	}

	void addVetoTruth(const TString& ct){
		vetotruths_.push_back(ct);
	}

	void addTagProbability(const TString& p){
		probabilities_.push_back(p);
	}

	void setInvalidCuts(const TString& p){
	    makeinvalidif_=p;
    }

	void setCuts(const TString& cut){
		cuts_=cut;
	}

	void setNBins(const size_t& nbins){
		nbins_=nbins;
	}

	void setLine(int col,int width=1,int style=1){
		linecol_=col;
		linewidth_=width;
		linestyle_=style;
	}
	void setLineWidth(int width=1){
		linewidth_=width;
	}

	const TString& name()const{return name_;}
     TString compatName()const{
        TString namecp=name_;
        namecp.ReplaceAll(" ","_");
        namecp.ReplaceAll("!","_");
        namecp.ReplaceAll("/","_");
        return namecp;}

	//now done in a simple tree-Draw way - if optmisation needed: switch to putting rocs in a loop (TBI)
	//would need a differnet way of implementing cuts
	void process(TChain *,std::ostream& out=std::cout);


	TGraph* getROC(){return &roc_;}

	const TH1D* getProbHisto()const{return &probh_;}
	const TH1D* getVetoProbHisto()const{return &vetoh_;}
    const TH1D* getInvalidatedHisto()const{return &invalidate_;}
    const TH1D* getInvalidatedVetoHisto()const{return &invalidate_veto_;}


private:

    static size_t nrocsCounter;
	size_t nbins_;

	TString name_;

	TString cuts_;

	std::vector<TString> truths_;
	std::vector<TString> vetotruths_;

	std::vector<TString> probabilities_;
	TString makeinvalidif_;

	TH1D probh_;
	TH1D vetoh_;
    TH1D invalidate_,invalidate_veto_;

	TGraph roc_;
	int linecol_,linewidth_,linestyle_;

	bool fullanalysis_;

};



#endif /* DEEPJET_ANALYSIS_INTERFACE_ROCCURVE_H_ */
