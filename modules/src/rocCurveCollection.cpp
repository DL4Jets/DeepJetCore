/*
 * rocCurveCollection.cpp
 *
 *  Created on: 24 Mar 2017
 *      Author: jkiesele
 */


#include "rocCurveCollection.h"
#include "colorToTColor.h"
#include "rocCurve.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TH1D.h"

void rocCurveCollection::addROC(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, int linecol, const TString& cuts, int linestyle){

	roccurves_.push_back(rocCurve(name,probability,truth,vetotruth,linecol,linestyle,cuts));

}

void rocCurveCollection::addROC(const TString& name, const TString& probability, const TString& truth,
		const TString& vetotruth, const TString& linecol, const TString& cuts, int linestyle){

	roccurves_.push_back(rocCurve(name,probability,truth,vetotruth,colorToTColor(linecol),linestyle,cuts));

}


void rocCurveCollection::printRocs(TChain* c, const TString& outpdf,const TString&outfile){

	gROOT->SetBatch();


	for(auto& rc:roccurves_){
		rc.process(c);
	}

	TCanvas cv;
	cv.SetGrid();

	TH1D haxis=TH1D("","",10,0,1);
	//haxis.Draw("AXIS");
	haxis.GetYaxis()->SetRangeUser(1e-3,1);
	//haxis.GetYaxis()->SetNdivisions(510);
	haxis.GetYaxis()->SetTitle("background efficiency");
	haxis.GetXaxis()->SetTitle("signal efficiency");
	//haxis.GetXaxis()->SetNdivisions(510);
	haxis.Draw("AXIS");
	haxis.Draw("AXIG,same");




	cv.cd();
	cv.SetLogy();
	gStyle->SetOptStat(0);

	leg_=new TLegend(0.15,0.5,0.4,0.85);
	leg_->SetBorderSize(1);
	leg_->SetFillColor(0);

	for(auto& rc:roccurves_){
		rc.setLineWidth(linewidth_);
		TGraph* g=rc.getROC();
		g->SetLineWidth(linewidth_);
		g->Draw("L,same");
		leg_->AddEntry(g,g->GetTitle(),"l");
	}

	leg_->Draw("same");
	//cv.RedrawAxis();
	cv.Print(outpdf);

}


//int linewidth_;
//std::vector<rocCurve> roccurves_;
