/*
 * rocCurve.cpp
 *
 *  Created on: 24 Mar 2017
 *      Author: jkiesele
 */




//	TString cuts_;
//
//	std::vector<TString> truths_;
//	std::vector<TString> vetotruths_;
//
//	std::vector<TString> probabilities_;
//
//	TH1D probh_;
//	TH1D vetoh_;
//
//	TGraph roc_;
//
//	bool firstcall_;

#include "rocCurve.h"
#include <iostream>
#include "TCanvas.h"

rocCurve::rocCurve():nbins_(100),linecol_(kBlack),linewidth_(1),linestyle_(1){

}
rocCurve::rocCurve(const TString& name):nbins_(100),linecol_(kBlack),linewidth_(1),linestyle_(1){
	name_=name;
}
rocCurve::rocCurve(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, const TString& cuts):nbins_(100),linecol_(kBlack),linewidth_(1),linestyle_(1){
	name_=name;
	addTagProbability(probability);
	addTruth(truth);
	addVetoTruth(vetotruth);
	setCuts(cuts);
}
rocCurve::rocCurve(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, int linecol, int linestyle, const TString& cuts)
:nbins_(100),linecol_(linecol),linewidth_(1),linestyle_(linestyle)
{
	name_=name;
	addTagProbability(probability);
	addTruth(truth);
	addVetoTruth(vetotruth);
	setCuts(cuts);
}


rocCurve::~rocCurve(){
	//empty for now
}

//now done in a simple tree-Draw way - if optmisation needed: switch to putting rocs in a loop (TBI)
void rocCurve::process(TChain *c){



	TCanvas cv;//just a dummy
	probh_=TH1D("prob","prob",nbins_,0,1);
	vetoh_=TH1D("veto","veto",nbins_,0,1);

	TString truthstr="";
	for(const auto& s:truths_)
		truthstr+=s+"+";
	truthstr.Remove(truthstr.Length()-1);

	TString vetostr="";
	for(const auto& s:vetotruths_)
		vetostr+=s+"+";
	vetostr.Remove(vetostr.Length()-1);

	TString probstr="";
	for(const auto& s:probabilities_)
		probstr+=s+"+";
	probstr.Remove(probstr.Length()-1);


	if(cuts_.Length()){
		c->Draw(probstr+">>prob",truthstr+"&&"+cuts_);
		c->Draw(probstr+">>veto",vetostr+"&&"+cuts_);
	}
	else{
		c->Draw(probstr+">>prob",truthstr);
		c->Draw(probstr+">>veto",vetostr);
	}


	//remove from mem list
	probh_.SetDirectory(0);
	vetoh_.SetDirectory(0);
	probh_.SetName("probh_");
	vetoh_.SetName("vetoh_");

	std::vector<double> p(nbins_),v(nbins_);

	double probintegral=probh_.Integral(0,100);
	double vetointegral=vetoh_.Integral(0,100);

	for(size_t i=0;i<nbins_;i++) {
		p[i] = probh_.Integral(i,100)/probintegral;
		v[i] = vetoh_.Integral(i,100)/vetointegral;
	}

	roc_=TGraph(nbins_,&p.at(0),&v.at(0));
	roc_.SetName(name_);
	roc_.SetTitle(name_);
	roc_.Draw("L");//necessary for some weird root reason
	roc_.SetLineColor(linecol_);
	roc_.SetLineStyle(linestyle_);
	roc_.SetLineWidth(linewidth_);
}







