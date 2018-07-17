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
#include "TFile.h"
#include "TLegendEntry.h"
#include "TLatex.h"
#include <fstream>

//void rocCurveCollection::addROC(const TString& name, const TString& probability, const TString& truth,
//        const TString& vetotruth, int linecol, const TString& cuts, int linestyle){
//
//    roccurves_.push_back(rocCurve(name,probability,truth,vetotruth,linecol,linestyle,cuts));
//
//}

void rocCurveCollection::addExtraLegendEntry(const TString& entr){
    extralegendtries_.push_back(entr);
}

void rocCurveCollection::addROC(const TString& name, const TString& probability, const TString& truth,
        const TString& vetotruth, const TString& linecol, const TString& cuts,const TString& invalidateif){

    rocCurve rc=rocCurve(name,probability,truth,vetotruth,colorToTColor(linecol),lineToTLineStyle(linecol),cuts,invalidateif);
    rc.setLineWidth(linewidth_);
    TString lc=linecol;
    lc.ToLower() ;
    if(lc.Contains("dummy"))
        rc.setLineWidth(0);
    roccurves_.push_back(rc);
    legentries_.push_back(name);
}


void rocCurveCollection::printRocs(TChain* c, const TString& outpdf,
        const TString&outfile, TCanvas* cv, TFile * f, std::vector<TChain*>* chainvec,double xmin_in){

    gROOT->SetBatch();

    TString filename=outpdf;
    filename=filename(0,filename.Length()-4);
    bool createFile=false;
    if(!f){
        f =new TFile(filename+".root","RECREATE");
        createFile=true;
    }
    size_t count=0;
    std::vector<TH1D*> probhistos,vetohistos,invalidhistos,invalidvetohistos;
    for(size_t i=0;i<roccurves_.size();i++){
        rocCurve& rc=roccurves_.at(i);
        std::ofstream outtxt((filename+"_"+rc.compatName()+".txt").Data());
        rc.setNBins(nbins_);
        if(c)
            rc.process(c,outtxt);
        else
            rc.process(chainvec->at(i),outtxt);
        TString tempname="tmph_";
        tempname+=count;
        count++;
        outtxt.close();

        TH1D* ha=(TH1D*)rc.getProbHisto()->Clone(tempname);
        probhistos.push_back(ha);
        tempname+=count;
        TH1D* hb=(TH1D*)rc.getVetoProbHisto()->Clone(tempname);
        vetohistos.push_back(hb);
        tempname+=count;
        TH1D* hc=(TH1D*)rc.getInvalidatedHisto()->Clone(tempname);
        invalidhistos.push_back(hc);
        tempname+=count;
        TH1D* hd=(TH1D*)rc.getInvalidatedHisto()->Clone(tempname);
        invalidvetohistos.push_back(hd);
    }


    bool createCanvas=false;
    if(!cv){
        cv=new TCanvas();
        createCanvas=true;
    }

    cv->SetGrid();
    cv->SetTicks(1, 1);
    cv->SetBottomMargin(0.12);
    cv->SetTopMargin(0.069);
    cv->SetLeftMargin(0.15);
    cv->SetRightMargin(0.03);
    cv->Draw();
    cv->cd();

    TH1D haxis=TH1D("AXIS","AXIS",10,0,1);
    //haxis.Draw("AXIS");
    haxis.GetYaxis()->SetRangeUser(8e-4,1);
    //haxis.GetYaxis()->SetNdivisions(510);

    haxis.GetYaxis()->SetTitleSize(0.05);
    //haxis.GetYaxis()->SetTitleOffset(0.9);
    haxis.GetYaxis()->SetLabelSize(0.045);
    if(xaxis_.Length())
        haxis.GetXaxis()->SetTitle(xaxis_);
    else
        haxis.GetXaxis()->SetTitle("efficiency");
    if(yaxis_.Length())
        haxis.GetYaxis()->SetTitle(yaxis_);
    else
        haxis.GetYaxis()->SetTitle("misid. probability");
    haxis.GetXaxis()->SetTitleSize(0.05);
    haxis.GetXaxis()->SetLabelSize(0.045);
    //haxis.GetXaxis()->SetTitleOffset(0.95);
    //haxis.GetXaxis()->SetRangeUser(0.4,1);
    //haxis.GetXaxis()->SetNdivisions(510);
    haxis.Draw("AXIS");
    haxis.Draw("AXIG,same");




    cv->cd();
    if(logy_)
        cv->SetLogy();
    gStyle->SetOptStat(0);
    float nreallegends=0;
    for(const auto& l:legentries_){
        if(l!="INVISIBLE" && l!="")
            nreallegends++;
    }

    leg_=new TLegend(0.18,0.72-(0.07*nreallegends),0.39,0.72);
    leg_->SetBorderSize(1);
    leg_->SetFillColor(0);

    double xmin=1;

    std::vector<TGraph*> graphs;
    count=0;

    for(size_t i=0;i<roccurves_.size();i++){
        TGraph* g=roccurves_.at(i).getROC();
        TString tempname="roccurve_";
        tempname+=i;
        g->SetName(tempname);
        graphs.push_back(g);
        g->SetLineWidth(linewidth_);
        for(double x=0;x<1;x+=0.01){
            double val=g->Eval(x);
            if(val>1e-2){
                if(xmin>x)
                    xmin=x;
                break;
            }
        }
        g->Draw("L,same");
        g->Write();
        if(legentries_.at(i)!="INVISIBLE" && legentries_.at(i)!="")
            leg_->AddEntry(g,legentries_.at(i),"l");
        vetohistos.at(count)->SetName((TString)g->GetName()+"_veto");
        vetohistos.at(count)->Write();
        probhistos.at(count)->SetName((TString)g->GetName()+"_prob");
        probhistos.at(count)->Write();
        invalidhistos.at(count)->SetName((TString)g->GetName()+"_invalid");
        invalidhistos.at(count)->Write();
        invalidvetohistos.at(count)->SetName((TString)g->GetName()+"_invalid_veto");
        invalidvetohistos.at(count)->Write();
        count++;
    }

    xmin*=10;
    xmin=(int)xmin;
    xmin/=10;
    if(xmin_in>0)
        haxis.GetXaxis()->SetRangeUser(xmin_in,1);
    else
        haxis.GetXaxis()->SetRangeUser(xmin,1);

    if(cmsstyle_){


        //add CMS labels
        TLatex *tex = new TLatex(0.18,0.865,"CMS Simulation");
        tex->SetNDC(true);
        tex->SetTextFont(61);
        tex->SetTextSize(0.08);
        tex->SetLineWidth(2);
        tex->Draw();

        tex = new TLatex(0.57,0.865,"#it{Preliminary}");
        tex->SetNDC(true);
        tex->SetTextFont(42);
        tex->SetTextSize(0.05);
        tex->SetLineWidth(2);
        tex->Draw();

        tex = new TLatex(.97,0.955,"#sqrt{s}=13 TeV, Phase 1");
        tex->SetNDC(true);
        tex->SetTextAlign(31);
        tex->SetTextFont(42);
        tex->SetTextSize(0.05);
        tex->SetLineWidth(2);
        tex->Draw();

        //haxis.GetXaxis()->SetTitle("b-jet efficiency");

        haxis.GetXaxis()->SetRangeUser(0,1);
    }
    leg_->Draw("same");
    ///////
    if(extralegendtries_.size()){
        TLegend* addleg=new TLegend(0.78,0.15,0.93,0.15+0.06*((float)( extralegendtries_.size() )));
        for(const auto& s:extralegendtries_){
            TString leg=s;
            if(! leg.Contains("?"))
                leg="solid?INVALIDINPUT";

            TString styledesc=leg(0, leg.First('?'));
            TString legentr=leg(leg.First('?')+1,leg.Length());


            TLegendEntry* e= addleg->AddEntry(legentr,legentr,"l");
            e->SetLineWidth(linewidth_);
            e->SetLineStyle(lineToTLineStyle(styledesc));
        }
        addleg->Draw("same");
    }
    //////


    for(auto& t:additionaltext_)
        t->Draw();
    //comment lines
    TLatex *tex = new TLatex(0.18,0.805,comment0_);
    tex->SetNDC(true);
    tex->SetTextFont(42);
    tex->SetTextSize(0.05);
    tex->SetLineWidth(2);
    tex->Draw();

    tex = new TLatex(0.18,0.7505,comment1_);
    tex->SetNDC(true);
    tex->SetTextFont(42);
    tex->SetTextSize(0.05);
    tex->SetLineWidth(2);
    tex->Draw();

    if(createCanvas){
        cv->Print(outpdf);
        cv->Write();
        cv->Print(filename+".C");

        //if(cmsstyle_) //not working due to missing libraries
        //    cv->Print(filename+".png");
    }
    if(createFile){
        f->Close();
        delete f;
    }

}


//int linewidth_;
//std::vector<rocCurve> roccurves_;
