//allows functions with 18 or less paramenters
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
//don't use new root6 stuff, has problems with leaf_list
//#include "TTreeReader.h"
//#include "TTreeReaderValue.h"
//#include "TTreeReaderArray.h"
//#include "TTreeReaderUtils.h"
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "TStopwatch.h"
#include "../interface/indata.h"
#include "../interface/pythonToSTL.h"
#include "../interface/helper.h"
#include <cmath>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>








/*
 * wrapper to create input to C++ only function
 * Can be generalised to doing it at the same time for many different sized branches
 */
void store(long numpyarray, const boost::python::list _shape,
        std::string filename) {
    std::vector<int> shape = toSTLVector<int>(_shape);    
    int nentries = shape[0];

    int ndims = shape.size() - 1;
    int shape_root[ndims];
    size_t flattened_length = 1;
    for(size_t i = 1; i < shape.size(); i++) {
        shape_root[i-1] = shape[i]; 
        flattened_length *= shape[i];
    }
    float data_root[flattened_length];
    float* data_in = reinterpret_cast<float*>(numpyarray);

    TFile *outfile = new TFile(filename.c_str(), "RECREATE");
    TDirectory *dir = outfile->mkdir("prediction", "prediction");
    dir->cd();
    TTree *t = new TTree("tree", "tree");

    t->Branch("ndims"  ,&ndims,   "ndims_/i");
    t->Branch("shape",   shape_root,   "shape_[ndims_]/i");
    t->Branch("flattened_length",   &flattened_length,   "flattened_length_/i");
    t->Branch("data",   data_root,   "data_[flattened_length_]/f");


    for(size_t e = 0; e < nentries; e++) {
        for (size_t i = 0; i < flattened_length; i++) {
            data_root[i]=data_in[e*flattened_length+i];
        }
        t->Fill();

    }

    t->Write();
    outfile->Close();
    delete outfile;


}








// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_storeTensor) {
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    __hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
    //anyway, it doesn't hurt, just leave this here
    def("store", &store);
}
