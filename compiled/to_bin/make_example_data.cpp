/*
 * make_example_data.cpp
 *
 *  Created on: 30 Apr 2019
 *      Author: jkiesele
 */

#include "TRandom3.h"
#include "TFile.h"
#include <vector>
#include "TTree.h"
#include "math.h"
#include <iostream>
#include <fstream>



class dataGenerator{
public:


    dataGenerator():size_(32),type_(0),rand_(new TRandom3(123)){setSize(size_);}
    ~dataGenerator(){delete rand_;}

    void gen();

    void setSize(int s){
        size_=s;
        image_.resize(size_,std::vector<float>(size_));

        xcoords_.clear();
        ycoords_.clear();
        for(float x=0;x<(float)size_;x++){
            for(float y=0;y<(float)size_;y++){
                ycoords_.push_back(y);
                xcoords_.push_back(x);
            }
        }

    }
    int getSize()const{return size_;}

    void setType(int t){
        type_=t;
    }

    const std::vector<float>& getXCoords()const{return xcoords_;}
    const std::vector<float>& getYCoords()const{return ycoords_;}

    const std::vector<std::vector<float> > & getImage()const{return image_;}

    std::vector<float>  getImageSeq()const;
    //not protected
       std::vector<std::vector<float> > addImage(const std::vector<std::vector<float> >&, const std::vector<std::vector<float> >&);

       std::vector<float>  addImage(const std::vector<float>&, const std::vector<float>&);


private:

    std::vector<std::vector<float> > makeImage(float xc, float yc, float xw, float yw)const;


    std::vector<std::vector<float> > image_;
    std::vector<float> xcoords_,ycoords_;
    int size_;
    int type_;
    TRandom3 * rand_;

};


int main(){

    dataGenerator gen;

    std::ofstream outtxtfile("files.txt");


    for(int i=0;i<10;i++){
        TString fname="out_";
        fname+=i;
        fname+=".root";

        outtxtfile << fname << std::endl;

        TFile f(fname,"RECREATE");
        TTree * t = new TTree("tree","tree");

        std::vector<float> imagetot;
        std::vector<float> * imagetotp = &imagetot;
        t->Branch("image",&imagetotp);
        std::vector<std::vector<float> > imagetot2d;
        std::vector<std::vector<float> > * imagetot2dp = &imagetot2d;
        t->Branch("image2d",&imagetot2dp);
        std::vector<float> xcoords;
        std::vector<float> * xcoordsp = &xcoords;
        t->Branch("xcoords",&xcoordsp);
        std::vector<float> ycoords;
        std::vector<float> * ycoordsp = &ycoords;
        t->Branch("ycoords",&ycoords);
        int size;
        t->Branch("size",&size);
        int isA,isB,isC;
        t->Branch("isA",&isA);
        t->Branch("isB",&isB);
        t->Branch("isC",&isC);
        size = gen.getSize();

        xcoords = gen.getXCoords();
        ycoords = gen.getYCoords();


        int type=0;

        for(size_t e=0;e<2000;e++){
            gen.setType(type);
            if(type==0){
                isA=1;isB=0;isC=0;
            }
            else if(type==1){
                isA=0;isB=1;isC=0;
            }
            else if(type==2){
                isA=0;isB=0;isC=1;
            }

            type++;
            if(type>2)
                type=0;


            gen.gen();
            auto s  = gen.getImageSeq();
            auto s2d = gen.getImage();
            gen.setType(3);
            gen.gen();
            auto bg = gen.getImageSeq();
            auto bg2d = gen.getImage();

            imagetot = gen.addImage(s,bg);

            imagetot2d = gen.addImage(s2d,bg2d);

            t ->Fill();
        }
        t->Write();
        f.Close();
        //delete t;
    }

    outtxtfile.close();
}





void dataGenerator::gen(){

    float xlow=0.45;
    float xhi = 0.55;
    float ylow = 0.45;
    float yhi = 0.55;
    float xw = 0.25*rand_->Uniform(0.95,1.05);
    float yw = 0.25*rand_->Uniform(0.95,1.05);
    //class 0

    if(type_==1){ //class 1
        xw = rand_->Uniform(0.1,0.2);
        yw = 0.5*xw*rand_->Uniform(0.95,1.05);

    }
    else if(type_==2){ //class 1
        yw = rand_->Uniform(0.1,0.3);
        xw = 1.8*yw*rand_->Uniform(0.95,1.05);

    }
    else if(type_>2){
       xlow=-5;
       xhi = 5;
       ylow = -5;
       yhi = 5;
       xw = rand_->Uniform(3,10);
       yw = rand_->Uniform(2,15);
    }

    float xc  = rand_->Uniform(xlow,xhi);
    float yc  = rand_->Uniform(ylow,yhi);


    image_ = makeImage(xc,yc,xw,yw);

}


std::vector<std::vector<float> > dataGenerator::makeImage(float xc, float yc, float xw, float yw)const{

    //to 'size' coordinates
    xc = (float)size_ * xc;
    yc = (float)size_ * yc;
    xw = (float)size_ * xw;
    yw = (float)size_ * yw;

    std::vector<std::vector<float> > out=image_;

    for(size_t x=0;x<out.size();x++){
        double dx = (float)x-xc;
        double xcontr = exp(-dx*dx/(2.*xw*xw));
        for(size_t y=0;y<out.size();y++){
            double dy = (float)y-yc;
            double ycontr = exp(-dy*dy/(2.*yw*yw));
            out.at(x).at(y) = xcontr*ycontr;
        }
    }

    return out;
}

std::vector<std::vector<float> > dataGenerator::addImage(const std::vector<std::vector<float> >& a, const std::vector<std::vector<float> >& b){
    auto  out = a;

    for(size_t i=0;i<a.size();i++)
        for(size_t j=0;j<a.size();j++)
            out.at(i).at(j) += b.at(i).at(j);
    return out;
}

std::vector<float>  dataGenerator::addImage(const std::vector<float>& a, const std::vector<float> & b){
    auto  out = a;

    for(size_t i=0;i<a.size();i++)
        out.at(i) += b.at(i);
    return out;
}



std::vector<float>  dataGenerator::getImageSeq()const{
    std::vector<float>  out;
    for(const auto& x:image_)
        for(const auto& y:x)
            out.push_back(y);
    return out;
}






