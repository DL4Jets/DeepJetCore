
#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
//#include "boost/filesystem.hpp"
#include <iostream>
#include <stdint.h>
#include "TString.h"
#include <string>
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "../interface/pythonToSTL.h"
#include "friendTreeInjector.h"
#include "TROOT.h"
#include "colorToTColor.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TFile.h"
#include "TStyle.h"
#include <algorithm>

using namespace boost::python; //for some reason....

static void mergeOverflow(TH1F*h){
    h->SetBinContent(h->GetNbinsX(),h->GetBinContent(h->GetNbinsX())+h->GetBinContent(h->GetNbinsX()+1));
    h->SetBinContent(1,h->GetBinContent(1)+h->GetBinContent(0));
}


void makePlots(
        const boost::python::list intextfiles,
        const boost::python::list names,
        const boost::python::list variables,
        const boost::python::list cuts,
        const boost::python::list colors,
        std::string outfile,
        std::string xaxis,
        std::string yaxis,
        bool normalized,
        bool makeProfile=false,
        bool makeWidthProfile=false,
        float OverrideMin=1e100,
        float OverrideMax=-1e100,
        std::string sourcetreename="deepntuplizer/tree",
        size_t nbins=0,
        float xmin=0,
        float xmax=0) {


    std::vector<TString>  s_intextfiles=toSTLVector<TString>(intextfiles);
    std::vector<TString>  s_vars = toSTLVector<TString>(variables);
    std::vector<TString>  s_names = toSTLVector<TString>(names);
    std::vector<TString>  s_colors = toSTLVector<TString>(colors);
    std::vector<TString>  s_cuts = toSTLVector<TString>(cuts);

    //reverse to make the first be on top
    std::reverse(s_intextfiles.begin(),s_intextfiles.end());
    std::reverse(s_vars.begin(),s_vars.end());
    std::reverse(s_names.begin(),s_names.end());
    std::reverse(s_colors.begin(),s_colors.end());
    std::reverse(s_cuts.begin(),s_cuts.end());

    TString toutfile=outfile;
    if(!toutfile.EndsWith(".pdf"))
        throw std::runtime_error("makePlots: output files need to be pdf format");


    if(!s_names.size())
        throw std::runtime_error("makePlots: needs at least one legend entry");
    /*
     * Size checks!!!
     */
    if(s_intextfiles.size() !=s_names.size()||
            s_names.size() != s_vars.size() ||
            s_names.size() != s_colors.size()||
            s_names.size() != s_cuts.size())
        throw std::runtime_error("makePlots: input lists must have same size");

    //make unique list of infiles
    std::vector<TString> u_infiles;
    std::vector<TString> aliases;
    TString oneinfile="";
    bool onlyonefile=true;
    for(const auto& f:s_intextfiles){
        if(oneinfile.Length()<1)
            oneinfile=f;
        else
            if(f!=oneinfile)
                onlyonefile=false;
    }
    for(const auto& f:s_intextfiles){
        //if(std::find(u_infiles.begin(),u_infiles.end(),f) == u_infiles.end()){
        u_infiles.push_back(f);
        TString s="";
        s+=aliases.size();
        aliases.push_back(s);
        //	std::cout << s <<std::endl;
        //}
    }



    friendTreeInjector injector(sourcetreename);
    for(size_t i=0;i<u_infiles.size();i++){
        if(!aliases.size())
            injector.addFromFile((TString)u_infiles.at(i));
        else
            injector.addFromFile((TString)u_infiles.at(i),aliases.at(i));
    }
    injector.createChain();

    TChain* c=injector.getChain();
    std::vector<TH1F*> allhistos;
    TLegend * leg=new TLegend(0.2,0.75,0.8,0.88);
    leg->SetBorderSize(0);

    leg->SetNColumns(3);
    leg->SetFillStyle(0);

    TString addstr="";
    if(normalized)
        addstr="normalized";
    if(makeProfile)
        addstr+="prof";
    else if(makeWidthProfile)
        addstr+="profs";
    if(makeProfile && makeWidthProfile)
        throw std::logic_error("makePlots: Not allowed to use makeProfile and makeWidthProfile at the same time");
    float max=-1e100;
    float min=1e100;

    TString tfileout=toutfile;
    tfileout=tfileout(0,tfileout.Length()-4);
    tfileout+=".root";

    TFile * f = new TFile(tfileout,"RECREATE");
    gStyle->SetOptStat(0);

    for(size_t i=0;i<s_names.size();i++){
        TString tmpname="hist_";
        tmpname+=i;
        TH1F *histo =0;
        if(nbins){
            histo = new TH1F(tmpname,tmpname,nbins,xmin,xmax);
        }

        c->Draw(s_vars.at(i)+">>"+tmpname,s_cuts.at(i),addstr);
        if(nbins<1){
            histo = (TH1F*) gROOT->FindObject(tmpname);
        }
        mergeOverflow(histo);
        histo->SetLineColor(colorToTColor(s_colors.at(i)));
        histo->SetLineStyle(lineToTLineStyle(s_colors.at(i)));
        histo->SetTitle(s_names.at(i));
        histo->SetName(s_names.at(i));

        histo->SetFillStyle(0);
        histo->SetLineWidth(2);



        float integral=histo->Integral("width");
        //the normalised option doesn't really do well
        if(integral && normalized)
            histo->Scale(1/integral);

        float tmax=histo->GetMaximum();
        float tmin=histo->GetMinimum();
        if(tmax>max)max=tmax;
        if(tmin<min)min=tmin;
        if((makeProfile||makeWidthProfile)  &&OverrideMin < OverrideMax){
            //std::cout << "overriding min/max"<< std::endl;
            max = OverrideMax;
            min = OverrideMin;
        }


        allhistos.push_back(histo);

        histo->Write();


    }
    for(size_t i=allhistos.size();i;i--){
        leg->AddEntry(allhistos.at(i-1),s_names.at(i-1),"l");
    }

    TCanvas cv("plots");

    allhistos.at(0)->Draw("AXIS");
    allhistos.at(0)->GetYaxis()->SetRangeUser(min,1.3*max); //space for legend on top

    allhistos.at(0)->GetXaxis()->SetTitle(xaxis.data());
    allhistos.at(0)->GetYaxis()->SetTitle(yaxis.data());

    allhistos.at(0)->Draw("AXIS");
    for(size_t i=0;i<s_names.size();i++){
        allhistos.at(i)->Draw("same,hist");
    }
    leg->Draw("same");

    cv.Write();
    cv.Print(toutfile);

    f->Close();


}


void makeEffPlots(
        const boost::python::list intextfiles,
        const boost::python::list names,
        const boost::python::list variables,
        const boost::python::list cutsnum,
        const boost::python::list cutsden,
        const boost::python::list colors,
        std::string outfile,
        std::string xaxis,
        std::string yaxis,
        int rebinfactor,
        bool setLogY,
	float Xmin,
	float Xmax,
        float OverrideMin=1e100,
        float OverrideMax=-1e100,
        std::string sourcetreename="deepntuplizer/tree"
		  )
  {


    std::vector<TString>  s_intextfiles=toSTLVector<TString>(intextfiles);
    std::vector<TString>  s_vars = toSTLVector<TString>(variables);
    std::vector<TString>  s_names = toSTLVector<TString>(names);
    std::vector<TString>  s_colors = toSTLVector<TString>(colors);
    std::vector<TString>  s_cutsnum = toSTLVector<TString>(cutsnum);
    std::vector<TString>  s_cutsden = toSTLVector<TString>(cutsden);

    //reverse to make the first be on top
    std::reverse(s_intextfiles.begin(),s_intextfiles.end());
    std::reverse(s_vars.begin(),s_vars.end());
    std::reverse(s_names.begin(),s_names.end());
    std::reverse(s_colors.begin(),s_colors.end());
    std::reverse(s_cutsnum.begin(),s_cutsnum.end());
    std::reverse(s_cutsden.begin(),s_cutsden.end());

    TString toutfile=outfile;
    if(!toutfile.EndsWith(".pdf"))
        throw std::runtime_error("makePlots: output files need to be pdf format");


    if(!s_names.size())
        throw std::runtime_error("makePlots: needs at least one legend entry");
    /*
     * Size checks!!!
     */
    if(s_intextfiles.size() !=s_names.size()||
            s_names.size() != s_vars.size() ||
            s_names.size() != s_colors.size()||
            s_names.size() != s_cutsden.size()||
            s_names.size() != s_cutsnum.size())
        throw std::runtime_error("makePlots: input lists must have same size");

    //make unique list of infiles
    std::vector<TString> u_infiles;
    std::vector<TString> aliases;
    TString oneinfile="";
    bool onlyonefile=true;
    for(const auto& f:s_intextfiles){
        if(oneinfile.Length()<1)
            oneinfile=f;
        else
            if(f!=oneinfile)
                onlyonefile=false;
    }
    for(const auto& f:s_intextfiles){
        //if(std::find(u_infiles.begin(),u_infiles.end(),f) == u_infiles.end()){
        u_infiles.push_back(f);
        TString s="";
        s+=aliases.size();
        aliases.push_back(s);
        //  std::cout << s <<std::endl;
        //}
    }

    friendTreeInjector injector(sourcetreename);
    for(size_t i=0;i<u_infiles.size();i++){
        if(!aliases.size())
            injector.addFromFile((TString)u_infiles.at(i));
        else
            injector.addFromFile((TString)u_infiles.at(i),aliases.at(i));
    }
    injector.createChain();

    TChain* c=injector.getChain();
    std::vector<TH1F*> allhistos;
    TLegend * leg=new TLegend(0.2,0.75,0.8,0.88);
    leg->SetBorderSize(0);

    leg->SetNColumns(3);
    leg->SetFillStyle(0);

    TString addstr="";
    float max=-1e100;
    float min=1e100;

    TString tfileout=toutfile;
    tfileout=tfileout(0,tfileout.Length()-4);
    tfileout+=".root";

    TFile * f = new TFile(tfileout,"RECREATE");
    gStyle->SetOptStat(0);

    for(size_t i=0;i<s_names.size();i++){
        TString tmpname="hist_";
        TString numcuts=s_cutsnum.at(i);
        if(s_cutsden.at(i).Length())
            numcuts+="&&("+s_cutsden.at(i)+")";
        tmpname+=i;
        c->Draw(s_vars.at(i)+">>"+tmpname,numcuts,addstr);
        TH1F *numhisto = (TH1F*) gROOT->FindObject(tmpname);
        if(rebinfactor>1)
            numhisto->Rebin(rebinfactor);
        TH1F *denhisto=(TH1F *)numhisto->Clone(tmpname+"den");


        c->Draw(s_vars.at(i)+">>"+tmpname+"den",s_cutsden.at(i),addstr);
        for(int bin=0;bin<=numhisto->GetNbinsX();bin++){
            float denbin=denhisto->GetBinContent(bin);
            if(denbin){
                numhisto->SetBinContent(bin, numhisto->GetBinContent(bin)/denbin);
            }
            else{
                numhisto->SetBinContent(bin,0);
            }
        }
        TH1F *histo = numhisto; //(TH1F *)()->Clone(tmpname) ;


        histo->SetLineColor(colorToTColor(s_colors.at(i)));
        histo->SetLineStyle(lineToTLineStyle(s_colors.at(i)));
        histo->SetTitle(s_names.at(i));
        histo->SetName(s_names.at(i));

        histo->SetFillStyle(0);
        histo->SetLineWidth(2);

        float tmax=histo->GetMaximum();
        float tmin=histo->GetMinimum();
        if(tmax>max)max=tmax;
        if(tmin<min)min=tmin;
        if(OverrideMin<OverrideMax){
            //std::cout << "overriding min/max"<< std::endl;
            max = OverrideMax;
            min = OverrideMin;
        }
        //std::cout << "min" << min << " max" << max << std::endl;

        allhistos.push_back(histo);

        histo->Write();


    }
    for(size_t i=allhistos.size();i;i--){
        leg->AddEntry(allhistos.at(i-1),s_names.at(i-1),"l");
    }

    TCanvas cv("plots");

    if(setLogY) cv.SetLogy();
    allhistos.at(0)->Draw("AXIS");
    allhistos.at(0)->GetYaxis()->SetRangeUser(min,1.3*max); //space for legend on top

    allhistos.at(0)->GetXaxis()->SetTitle(xaxis.data());
    if(Xmin<Xmax)  allhistos.at(0)->GetXaxis()->SetRangeUser(Xmin,Xmax);
    allhistos.at(0)->GetYaxis()->SetTitle(yaxis.data());

    allhistos.at(0)->Draw("AXIS");
    for(size_t i=0;i<s_names.size();i++){
        allhistos.at(i)->Draw("same,hist");
    }
    leg->Draw("same");

    cv.Write();
    cv.Print(toutfile);

    f->Close();


}

void makeProfiles(
        const boost::python::list intextfiles,
        const boost::python::list names,
        const boost::python::list variables,
        const boost::python::list cuts,
        const boost::python::list colors,
        std::string outfile,
        std::string xaxis,
        std::string yaxis,
        bool normalized,float minimum,
        float maximum,
        std::string treename) {

    makePlots(
            intextfiles,
            names,
            variables,
            cuts,
            colors,
            outfile,
            xaxis,
            yaxis,
            normalized,
            true,false,
            minimum,
            maximum,treename);
}  



// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_makePlots) {
    //__hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
    //anyway, it doesn't hurt, just leave this here
    def("makePlots", &makePlots);
    def("makeEffPlots", &makeEffPlots);
    def("makeProfiles", &makeProfiles);

}

