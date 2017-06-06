/*
 * dataSizePlots.C
 *
 *  Created on: 16 May 2017
 *      Author: jkiesele
 */


#include "TGraph.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TLegend.h"

#include "fstream"
#include <string>
class fullscan{
public:
    fullscan(std::string infile){
        readFromFile(infile);
    }
    void readFromFile(std::string infilename){
        x_.clear();y_.clear();
        std::ifstream infile(infilename);
        double a,b;
        double c=0;
        while (infile >> a >> b){
            x_.push_back(c);
            y_.push_back(b);
            c++;
        }

    }

    void cutFirst(size_t ncut){
        if((int)ncut>N())return;
        x_= std::vector<double>(x_.begin()+ncut,x_.end());
        y_= std::vector<double>(y_.begin()+ncut,y_.end());
    }

    const double * getX()const{
        return &x_.at(0);
    }
    const double * getY()const{
        return &y_.at(0);
    }
    int N()const{
        return x_.size();
    }

    TGraph* getGraph(TString title, int color)const{
        TGraph* epdep=new TGraph(N(),getX(),getY());
        epdep->GetYaxis()->SetTitle("val loss");
        epdep->GetXaxis()->SetTitle("#epochs");
        epdep->SetLineColor(color);
        epdep->SetLineWidth(2);
        return epdep;
    }
    TGraph* getGraph(TString title, int color,TLegend* leg)const{
        TGraph* epdep=getGraph(title,color);
        leg->AddEntry(epdep,title,"l");
        return epdep;
    }

private:
    std::vector<double> x_,y_;

};

class scanPoint{
public:
    scanPoint(double Njets, double Nepochs, double Valloss){
        njets=Njets;nepochs=Nepochs;valloss=Valloss;
    }
    double njets,nepochs,valloss;

    operator double()const{
        return valloss;
    }
};

void applyStyle(TGraph* g){
    g->SetLineWidth(2);

}



void dataSizePlots(){


    TCanvas cv;
    gStyle->SetOptTitle(0);
    cv.SetBottomMargin(.15);
    cv.SetLeftMargin(.25);
    //create the data

    /*
    const scanPoint scan_10_10(10,10,0.495591667195);
    const scanPoint scan_10_20(10,20,0.487303707828);
    const scanPoint scan_10_40(10,40,0.483191596489);
    const scanPoint scan_20_10(20,10,0.483344945623);
    const scanPoint scan_20_20(20,20,0.476566161458);
    const scanPoint scan_20_40(20,40,0.474743667435);
    const scanPoint scan_35_10(35,10,0.483023190782);
    const scanPoint scan_35_20(35,20,0.472163896581);
    const scanPoint scan_35_40(35,40,0.46094467469);
    const scanPoint scan_40_10(40,10,0.475578535775);
    const scanPoint scan_40_20(40,20,0.471653718445);
    const scanPoint scan_40_40(40,40,0.459034395315);

    TLegend * leg=new TLegend(0.3,0.2,.55,0.45);

    double epdep_y[]={scan_10_10,scan_10_20,scan_10_40};
    double epdep_x[]={10,20,40};

    TGraph* epdep=new TGraph(3,epdep_x,epdep_y);
    epdep->GetYaxis()->SetTitle("val loss");
    epdep->GetXaxis()->SetTitle("#epochs");
    leg->AddEntry(epdep,"10 M jets","l");

    epdep->Draw("AL");
    epdep->GetYaxis()->SetRangeUser(0.445,0.5);
    epdep->SetLineColor(kBlue);
    applyStyle(epdep);

    double epdep2_y[]={scan_20_10,scan_20_20,scan_20_40};
    double epdep2_x[]={10,20,40};

    TGraph* epdep2=new TGraph(3,epdep2_x,epdep2_y);
    epdep2->SetLineColor(kMagenta);
    applyStyle(epdep2);
    epdep2->Draw("L,same");
    leg->AddEntry(epdep2,"20 M jets","l");


    double epdep3_y[]={scan_35_10,scan_35_20,scan_35_40};
    double epdep3_x[]={10,20,40};

    TGraph* epdep3=new TGraph(3,epdep3_x,epdep3_y);
    epdep3->SetLineColor(kBlack);
    applyStyle(epdep3);
    epdep3->Draw("L,same");
    leg->AddEntry(epdep3,"35 M jets","l");


    double epdep4_y[]={scan_40_10,scan_40_20,scan_40_40};
    double epdep4_x[]={10,20,40};

    TGraph* epdep4=new TGraph(3,epdep4_x,epdep4_y);
    epdep4->SetLineColor(kRed);
    applyStyle(epdep4);
    epdep4->Draw("L,same");
    leg->AddEntry(epdep4,"40 M jets","l");

    leg->Draw("same");
     */

    //read in data



    fullscan jets40("DF_map_40M/losses.log");
    fullscan jets80("DF_map_80M/losses.log");



    TLegend * leg=new TLegend(0.3,0.2,.55,0.45);
    leg->SetFillStyle(0);

    TGraph* epdep=jets40.getGraph("40 M jets",kBlue,leg);
    epdep->Draw("AL");
    epdep->GetYaxis()->SetRangeUser(1.02,1.09);


   // TGraph* epdep2=jets20.getGraph("20 M jets",kMagenta,leg);
   // epdep2->Draw("L,same");


    TGraph* epdep3=jets80.getGraph("80 M jets",kBlack,leg);
    epdep3->Draw("L,same");




    leg->Draw("same");

    cv.Print("epdep.pdf");





}
