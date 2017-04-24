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
        const TString& vetotruth, int linecol, int linestyle, const TString& cuts,const TString& invalidateif)
:nbins_(100),linecol_(linecol),linewidth_(1),linestyle_(linestyle)
{
    name_=name;
    addTagProbability(probability);
    addTruth(truth);
    addVetoTruth(vetotruth);
    setCuts(cuts);
    setInvalidCuts(invalidateif);
}


rocCurve::~rocCurve(){
    //empty for now
}

//now done in a simple tree-Draw way - if optmisation needed: switch to putting rocs in a loop (TBI)
void rocCurve::process(TChain *c){




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

    TString allcuts=truthstr;
    TString allinvalid=makeinvalidif_;
    if(allinvalid.Length()<1){
        allinvalid=probstr+"<-10000"; //false
    }

    allinvalid+="&&"+truthstr;

    if(cuts_.Length()){
        if(allcuts.Length())
            allcuts=truthstr+"&&"+cuts_;
        else
            allcuts=cuts_;
        if(vetostr.Length())
            vetostr=vetostr+"&&"+cuts_;
        else
            vetostr=cuts_;

        allinvalid=allinvalid+"&&"+cuts_;
    }




    TCanvas cv;//just a dummy
    probh_=TH1D("prob","prob",nbins_,0,1);
    vetoh_=TH1D("veto","veto",nbins_,0,1);
    invalidate_=TH1D("invalid","invalid",nbins_,0,1);


    c->Draw(probstr+">>prob",allcuts);//probcuts);
    c->Draw(probstr+">>veto",vetostr);
    c->Draw(probstr+">>invalid",allinvalid);


    //remove from mem list
    probh_.SetDirectory(0);
    vetoh_.SetDirectory(0);
    invalidate_.SetDirectory(0);

    probh_.SetName("probh_");
    vetoh_.SetName("vetoh_");
    invalidate_.SetName("invalidate_");

    std::vector<double> p(nbins_),v(nbins_);

    double probintegral=probh_.Integral(0,nbins_);
    double vetointegral=vetoh_.Integral(0,nbins_);
   // double invalidintegral=invalid.Integral(0,nbins_);

    probh_.Add(&invalidate_,-1.);
    for(int i=0;i<=probh_.GetNbinsX();i++){
        if(probh_.GetBinContent(i)<0)probh_.SetBinContent(i,0);//just safety measure
    }

    for(size_t i=0;i<nbins_;i++) {
        p[i] = probh_.Integral(i,nbins_)/(probintegral);
        v[i] = vetoh_.Integral(i,nbins_)/vetointegral;
    }

    roc_=TGraph(nbins_,&p.at(0),&v.at(0));
    roc_.SetName(name_);
    roc_.SetTitle(name_);
    roc_.Draw("L");//necessary for some weird root reason
    roc_.SetLineColor(linecol_);
    roc_.SetLineStyle(linestyle_);
    roc_.SetLineWidth(linewidth_);
}







