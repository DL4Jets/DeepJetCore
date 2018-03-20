/*
 * rocCurveCollection.h
 *
 *  Created on: 24 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_ANALYSIS_INTERFACE_ROCCURVECOLLECTION_H_
#define DEEPJET_ANALYSIS_INTERFACE_ROCCURVECOLLECTION_H_

#include "rocCurve.h"
#include <vector>
#include "TLegend.h"
#include "TChain.h"
#include "TFile.h"
#include "TCanvas.h"
#include <vector>
class TLatex;

class rocCurveCollection{
public:
	rocCurveCollection():leg_(0),linewidth_(2),cmsstyle_(false),logy_(true),nbins_(100){}
	~rocCurveCollection(){
		if(leg_)
			delete leg_;
		leg_=0;
	}

	void setLineWidth(int width){
		linewidth_=width;
	}

	void setCommentLine0(const TString& l){
	    comment0_=l;
	}

    void setCommentLine1(const TString& l){
        comment1_=l;
    }

    void setNBins(size_t nbins){
        nbins_=nbins;
    }

    void addExtraLegendEntry(const TString& entr);

	void setCMSStyle(bool cmsst){cmsstyle_=cmsst;}
	void setLogY(bool logy){logy_=logy;}
	void setXaxis(TString axis){xaxis_=axis;}
    void setYaxis(TString axis){yaxis_=axis;}

//	void addROC(const TString& name, const TString& probability, const TString& truth,
//		const TString& vetotruth, int linecolstyle, const TString& cuts="",int linestyle=1);

	void addROC(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, const TString& linecolstyle, const TString& cuts="",const TString& invalidateif="");

	void addText(TLatex *l){additionaltext_.push_back(l);}

	void printRocs(TChain* c, const TString& outpdf,const TString&outfile="",TCanvas* cv=0, TFile * f=0,
	        std::vector<TChain*>* chainvec=0,double xmin_in=-1);

private:
	TLegend * leg_;
	int linewidth_;
	std::vector<rocCurve> roccurves_;
	std::vector<TString> legentries_;
    std::vector<TString> extralegendtries_;
	bool cmsstyle_;
	TString comment0_, comment1_;
	bool logy_;
	TString xaxis_,yaxis_;
	size_t nbins_;
	std::vector<TLatex *> additionaltext_;
};


#endif /* DEEPJET_ANALYSIS_INTERFACE_ROCCURVECOLLECTION_H_ */
