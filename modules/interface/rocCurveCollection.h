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

class rocCurveCollection{
public:
	rocCurveCollection():leg_(0),linewidth_(2){}
	~rocCurveCollection(){
		if(leg_)
			delete leg_;
		leg_=0;
	}

	void setLineWidth(int width){
		linewidth_=width;
	}

	void addROC(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, int linecol, const TString& cuts="", int linestyle=1);

	void addROC(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, const TString& linecol, const TString& cuts="", int linestyle=1);


	void printRocs(TChain* c, const TString& outpdf,const TString&outfile="");

private:
	TLegend * leg_;
	int linewidth_;
	std::vector<rocCurve> roccurves_;
};


#endif /* DEEPJET_ANALYSIS_INTERFACE_ROCCURVECOLLECTION_H_ */
