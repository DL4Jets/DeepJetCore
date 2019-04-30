
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



void read2DArray(boost::python::numeric::array numpyarray,
        std::string filename_std,
        std::string treename_std,
        std::string branchname_std) {


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

    if(!nentries || nentries != tree->GetEntries())
        throw std::runtime_error("read2DArray: tree/array entries don't match");


    std::vector<std::vector<float> > *inarr = 0;
    tree->SetBranchAddress(branchname_std.data(),&inarr);

    tree->GetEntry(0);

    if(nx!=(int)inarr->size() || ny!=(int)inarr->at(0).size())
        throw std::runtime_error("read2DArray: tree/array dimensions don't match");

    for(int e=0;e<nentries;e++){
        tree->GetEntry(e);
        for(size_t x=0;x<inarr->size();x++)
            for(size_t y=0;y<inarr->at(x).size();y++)
                numpyarray[e][x][y][0] = inarr->at(x)[y];
    }
    tfile->Close();
    delete tfile;
}


void read3DArray(boost::python::numeric::array numpyarray,
        std::string filename_std,
        std::string treename_std,
        std::string branchname_std) {


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

    if(!nentries || nentries != tree->GetEntries())
        throw std::runtime_error("read3DArray: tree/array entries don't match");


    std::vector<std::vector<std::vector<float> > > * inarr = 0;
    tree->SetBranchAddress(branchname_std.data(),&inarr);

    tree->GetEntry(0);

    if(nx!=(int)inarr->size() || ny!=(int)inarr->at(0).size())
        throw std::runtime_error("read3DArray: tree/array dimensions don't match");

    for(int e=0;e<nentries;e++){
        tree->GetEntry(e);
        for(size_t x=0;x<inarr->size();x++)
            for(size_t y=0;y<inarr->at(x).size();y++)
                for(size_t z=0;z<inarr->at(x)[y].size();z++)
                    numpyarray[e][x][y][z][0] = inarr->at(x)[y][z];
    }
    tfile->Close();
    delete tfile;
}


// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_arrayReads) {
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    def("read2DArray", &read2DArray);
    def("read3DArray", &read3DArray);
}

