
#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
//#include "boost/filesystem.hpp"
#include <iostream>
#include <stdint.h>
#include "TString.h"
#include <string>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "TStopwatch.h"
#include "../interface/indata.h"
#include "../interface/pythonToSTL.h"
#include "../interface/helper.h"
#include <cmath>


using namespace boost::python; //for some reason....



void read2DArray(boost::python::numpy::ndarray numpyarray,
        std::string filename_std,
        std::string treename_std,
        std::string branchname_std,
        int rebinx,
        int rebiny,
        bool zeropad,
        bool x_cutoff,
        boost::python::numpy::ndarray x_ncut
        ) {


    TFile * tfile = new TFile(filename_std.data(),"READ");
    checkTObject(tfile,"read2DArray: input file problem");

    TTree* tree=(TTree*)tfile->Get(treename_std.data());
    checkTObject(tree,"read2DArray: input tree problem");


    int nentries = (int) boost::python::len(numpyarray);
    int nx=0;
    if(nentries)
        nx = (int) boost::python::len(numpyarray[0]);
    int ny=0;
    if(nx)
        ny = (int) boost::python::len(numpyarray[0][0]);

    if(!nentries || nentries != tree->GetEntries()){
        std::cerr << "read2DArray: tree/array entries don't match" << std::endl;
        throw std::runtime_error("read2DArray: tree/array entries don't match");
    }


    std::vector<std::vector<float> > *inarr = 0;
    tree->SetBranchAddress(branchname_std.data(),&inarr);

    tree->GetEntry(0);

    if(!zeropad && (nx*rebinx!=(int)inarr->size() || ny*rebiny!=(int)inarr->at(0).size())){
        std::cerr << "read2DArray: tree/array dimensions don't match" << std::endl;
        throw std::runtime_error("read2DArray: tree/array dimensions don't match");
    }

    int npe=0;
    for(int e=0;e<nentries;e++){
        tree->GetEntry(e);
        if(inarr->size() > nx){
            if(x_cutoff){
                x_ncut[0]+=1;
                continue;}
            else throw std::out_of_range("read2DArray: x ([:,x,...]) out of range");
        }
        for(size_t x=0;x<inarr->size();x++){
            int npx = (int)x/rebinx;
            for(size_t y=0;y<inarr->at(x).size();y++){
                int npy = (int)y/rebiny;
                numpyarray[npe][npx][npy][0] += inarr->at(x)[y];
            }
        }
        npe++;
    }
    tfile->Close();
    delete tfile;
}


void read3DArray(boost::python::numpy::ndarray numpyarray,
        std::string filename_std,
        std::string treename_std,
        std::string branchname_std,
        int rebinx=1,
        int rebiny=1,
        int rebinz=1,
        bool zeropad=false) {


    TFile * tfile = new TFile(filename_std.data(),"READ");
    checkTObject(tfile,"read2DArray: input file problem");

    TTree* tree=(TTree*)tfile->Get(treename_std.data());
    checkTObject(tree,"read2DArray: input tree problem");


    int nentries = (int) boost::python::len(numpyarray);
    int nx=0;
    if(nentries)
        nx = (int) boost::python::len(numpyarray[0]);
    int ny=0;
    if(nx)
        ny = (int) boost::python::len(numpyarray[0][0]);
    int nz=0;
    if(ny)
        nz = (int) boost::python::len(numpyarray[0][0][0]);

    if(!nentries || nentries != tree->GetEntries()){
        std::cerr << "read3DArray: tree/array entries don't match" << std::endl;
        throw std::runtime_error("read3DArray: tree/array entries don't match");
    }


    std::vector<std::vector<std::vector<float> > > * inarr = 0;
    tree->SetBranchAddress(branchname_std.data(),&inarr);

    tree->GetEntry(0);

    if(!zeropad && (nx*rebinx!=(int)inarr->size() || ny*rebiny!=(int)inarr->at(0).size() || nz*rebinz!=(int)inarr->at(0).at(0).size())){
        std::cerr << "read3DArray: tree/array dimensions don't match" << std::endl;
        throw std::runtime_error("read3DArray: tree/array dimensions don't match");
    }

    for(int e=0;e<nentries;e++){
        tree->GetEntry(e);
        for(size_t x=0;x<inarr->size();x++){
            int npx = (int)x/rebinx;
            for(size_t y=0;y<inarr->at(x).size();y++){
                int npy = (int)y/rebiny;
                for(size_t z=0;z<inarr->at(x)[y].size();z++){
                    int npz = (int)z/rebinz;
                    numpyarray[e][npx][npy][npz][0] += inarr->at(x)[y][z];
                }
            }
        }
    }
    tfile->Close();
    delete tfile;
}


void read4DArray(boost::python::numpy::ndarray numpyarray,
        std::string filename_std,
        std::string treename_std,
        std::string branchname_std,
        int rebinx=1,
        int rebiny=1,
        int rebinz=1,
        int rebinf=1,
        bool zeropad=false) {


    TFile * tfile = new TFile(filename_std.data(),"READ");
    checkTObject(tfile,"read2DArray: input file problem");

    TTree* tree=(TTree*)tfile->Get(treename_std.data());
    checkTObject(tree,"read2DArray: input tree problem");


    int nentries = (int) boost::python::len(numpyarray);
    int nx=0;
    if(nentries)
        nx = (int) boost::python::len(numpyarray[0]);
    int ny=0;
    if(nx)
        ny = (int) boost::python::len(numpyarray[0][0]);
    int nz=0;
    if(ny)
        nz = (int) boost::python::len(numpyarray[0][0][0]);
    int nf=0;
    if(nz)
        nf = (int) boost::python::len(numpyarray[0][0][0][0]);

    if(!nentries || nentries != tree->GetEntries()){
        std::cerr << "read4DArray: tree/array entries don't match" << std::endl;
        throw std::runtime_error("read4DArray: tree/array entries don't match");
    }

    std::vector<std::vector<std::vector<std::vector<float> > > > * inarr = 0;
    tree->SetBranchAddress(branchname_std.data(),&inarr);

    tree->GetEntry(0);

    if(!zeropad && (nx*rebinx!=(int)inarr->size() || ny*rebiny!=(int)inarr->at(0).size() || nz*rebinz!=(int)inarr->at(0).at(0).size()
            || nf*rebinf!=(int)inarr->at(0).at(0).at(0).size())){
        std::cout << "nx*rebinx "<<nx*rebinx<<", in "<< inarr->size()<<'\n';
        std::cout << "ny*rebiny "<<ny*rebiny<<", in "<< inarr->at(0).size()<<'\n';
        std::cout << "nz*rebinz "<<nz*rebinz<<", in "<< inarr->at(0).at(0).size()<<'\n';
        std::cout << "nf*rebinf "<<nf*rebinf<<", in "<< inarr->at(0).at(0).at(0).size()<<'\n';
        throw std::runtime_error("read4DArray: tree/array dimensions don't match");
    }
    for(int e=0;e<nentries;e++){
        tree->GetEntry(e);
        for(size_t x=0;x<inarr->size();x++){
            int npx = (int)x/rebinx;
            for(size_t y=0;y<inarr->at(x).size();y++){
                int npy = (int)y/rebiny;
                for(size_t z=0;z<inarr->at(x)[y].size();z++){
                    int npz = (int)z/rebinz;
                    for(size_t f=0;f<inarr->at(x)[y][z].size();f++){
                        int npf = (int)f/rebinf;
                       // std::cout << e <<", "<< npx <<", "<< npy <<", "<< npz <<", "<< npf << ": "<< f<< std::endl;
                        numpyarray[e][npx][npy][npz][npf][0] += inarr->at(x)[y][z][f];
                    }
                }
            }
        }
    }
    tfile->Close();
    delete tfile;
}


// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_arrayReads) {

    boost::python::numpy::initialize();
    def("read2DArray", &read2DArray);
    def("read3DArray", &read3DArray);
    def("read4DArray", &read4DArray);
}

