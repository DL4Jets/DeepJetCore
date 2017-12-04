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

size_t rocCurve::nrocsCounter=0;

static std::vector<double> loglist(double first, double last, double size){
    if(first>last) std::swap(first,last);
    double logfirst = log(first)/log(10);
    double loglast = log(last)/log(10);
    double step = (loglast-logfirst)/(size-1);
    std::vector<double> out;
    for(double x=logfirst; x<=loglast; x+=step)
    {
        double a = pow(10,x);
        out.push_back(a);
    }
    return out;

}


rocCurve::rocCurve():nbins_(100),linecol_(kBlack),linewidth_(1),linestyle_(1),fullanalysis_(true){
    nrocsCounter++;
}
rocCurve::rocCurve(const TString& name):nbins_(100),linecol_(kBlack),linewidth_(1),linestyle_(1),fullanalysis_(true){
    name_=name;
}
rocCurve::rocCurve(const TString& name, const TString& probability, const TString& truth,
        const TString& vetotruth, const TString& cuts):nbins_(100),linecol_(kBlack),linewidth_(1),linestyle_(1),fullanalysis_(true){
    name_=name;
    addTagProbability(probability);
    addTruth(truth);
    addVetoTruth(vetotruth);
    setCuts(cuts);
}
rocCurve::rocCurve(const TString& name, const TString& probability, const TString& truth,
        const TString& vetotruth, int linecol, int linestyle, const TString& cuts,const TString& invalidateif)
:nbins_(100),linecol_(linecol),linewidth_(1),linestyle_(linestyle),fullanalysis_(true)
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
void rocCurve::process(TChain *c,std::ostream& out){




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
    TString allinvalid_truth=makeinvalidif_;
    if(allinvalid_truth.Length()<1){
        allinvalid_truth=probstr+"<-10000"; //false
    }

    TString allinvalid_veto=allinvalid_truth+"&&"+vetostr;
    allinvalid_truth+="&&"+truthstr;

    if(cuts_.Length()){
        if(allcuts.Length())
            allcuts=truthstr+"&&"+cuts_;
        else
            allcuts=cuts_;
        if(vetostr.Length())
            vetostr=vetostr+"&&"+cuts_;
        else
            vetostr=cuts_;

        allinvalid_truth=allinvalid_truth+"&&"+cuts_;
        allinvalid_veto+="&&"+cuts_;
    }


    TString nrcc="";
    nrcc+=nrocsCounter;

    TCanvas cv;//just a dummy
    probh_=TH1D("prob"+nrcc,"prob"+nrcc,nbins_,0,1);
    vetoh_=TH1D("veto"+nrcc,"veto"+nrcc,nbins_,0,1);
    invalidate_=TH1D("invalid"+nrcc,"invalid"+nrcc,nbins_,0,1);
    invalidate_veto_=TH1D("invalid_veto"+nrcc,"invalid_veto"+nrcc,nbins_,0,1);


    c->Draw(probstr+">>prob"+nrcc,allcuts);//probcuts);
    c->Draw(probstr+">>veto"+nrcc,vetostr);
    c->Draw(probstr+">>invalid"+nrcc,allinvalid_truth);
    c->Draw(probstr+">>invalid_veto"+nrcc,allinvalid_veto);


    //remove from mem list
    probh_.SetDirectory(0);
    vetoh_.SetDirectory(0);
    invalidate_.SetDirectory(0);
    invalidate_veto_.SetDirectory(0);

    probh_.SetName("probh_"+nrcc);
    vetoh_.SetName("vetoh_"+nrcc);
    invalidate_.SetName("invalidate_"+nrcc);
    invalidate_veto_.SetName("invalidate_veto_"+nrcc);

    std::vector<double> p(nbins_),v(nbins_);

    double probintegral=probh_.Integral(0,nbins_);
    double vetointegral=vetoh_.Integral(0,nbins_);
    // double invalidintegral=invalid.Integral(0,nbins_);

    probh_.Add(&invalidate_,-1.);
    vetoh_.Add(&invalidate_veto_,-1.);

    for(int i=0;i<=probh_.GetNbinsX();i++){
        if(probh_.GetBinContent(i)<0)probh_.SetBinContent(i,0);//just safety measure
    }

    for(size_t i=0;i<nbins_;i++) {
        p[i] = probh_.Integral(i,nbins_)/(probintegral);
        v[i] = vetoh_.Integral(i,nbins_)/vetointegral;
    }
    TString compatname=name_;
    compatname.ReplaceAll(" ","_");
    compatname.ReplaceAll("/","_");
    compatname.ReplaceAll(":","_");
    compatname.ReplaceAll("!","_");

    roc_=TGraph(nbins_,&p.at(0),&v.at(0));
    roc_.SetName(compatname+nrcc);
    roc_.SetTitle(name_+nrcc);
    roc_.Draw("L");//necessary for some weird root reason


    out << "eff @ misid @ discr value\n\n";
    std::vector<double> misidset=loglist(0.001,1,100);
    int count=0;
    double integral=0;
    for(float eff=0;eff<1;eff+=0.00001){

        float misid=roc_.Eval(eff);
        integral+=misid*0.00001;
        if(misid>misidset[count]){
            out << eff <<"@"<< misid;
            count++;
            //search for closest bin
            if(fullanalysis_){
                for(size_t i=0;i<nbins_-1;i++) {
                    double integra=probh_.Integral(i,nbins_)/(probintegral);
                    double integrb=probh_.Integral(i+1,nbins_)/(probintegral);
                    if(eff>integrb && eff<integra){
                        out<< "@"<<probh_.GetBinCenter(i);
                    }
                }
            }
            out <<std::endl;
        }

    }
    out << "Area under ROC: "<< integral<<std::endl;
    roc_.SetLineColor(linecol_);
    roc_.SetLineStyle(linestyle_);
    roc_.SetLineWidth(linewidth_);
}







