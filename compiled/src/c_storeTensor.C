//allows functions with 18 or less paramenters
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
#include <vector>

#define Vec std::vector


float * create0DBranchVector(std::vector<int> shape){//returns pointer to first element
    return new float[1];
}


Vec<float> * create1DBranchVector(std::vector<int> shape ){//returns pointer to first element
    return new Vec<float>(shape[0]);
}

//gets called once anyway, still needs garbage collection
Vec<Vec<float> >  * create2DBranchVector(std::vector<int> shape){//returns pointer to first element
    return new Vec<Vec<float> >
    (shape[0],Vec<float>(shape[1]));
}

Vec<Vec<Vec<float> > >* create3DBranchVector(std::vector<int> shape){//returns pointer to first element
    return new Vec<Vec<Vec<float> > >
    (shape[0],Vec<Vec<float> > (shape[1],Vec<float>  (shape[2])));
}

Vec<Vec<Vec<Vec<float> > > > * create4DBranchVector(std::vector<int> shape){//returns pointer to first element
    auto * out = new
            Vec<Vec<Vec<Vec<float> > > >(shape[0],
                    Vec<Vec<Vec<float> > >(shape[1],
                            Vec<Vec<float> >(shape[2],
                                    Vec<float>(shape[3]))));
    for(auto& vvv:*out)
        for(auto& vv:vvv)
            for(auto & v:vv)
                for(auto& e:v)
                    e=0;
    return out;
}

/*
 * wrapper to create input to C++ only function
 * Can be generalised to doing it at the same time for many different sized branches
 */
void store(const boost::python::list _numpyarrays,
        const boost::python::list _shapes,
        std::string filename,
        const boost::python::list _adds,
        bool storeflat) {

    Vec<long> numpyarrays = toSTLVector<long>(_numpyarrays);
    Vec<Vec<int> > shapes = toSTL2DVector<int>(_shapes);
    Vec<std::string> adds = toSTLVector<std::string>(_adds);

    int nentries = 0;


    TFile *outfile = new TFile(filename.c_str(), "RECREATE");
    //TDirectory *dir = outfile->mkdir("prediction", "prediction");
    //dir->cd();
    TTree *t = new TTree("tree", "tree");

    //create branches and pointers
    Vec<float* >   v_data_in;
    Vec<Vec<int> > v_shape;
    Vec<int>       v_ndims;
    Vec<int>       v_flattened_length;
    Vec<float* >                         v_data_out_0D(numpyarrays.size(),0);
    Vec<Vec<float>* >                    v_data_out_1D(numpyarrays.size(),0);
    Vec<Vec<Vec<float> >* >              v_data_out_2D(numpyarrays.size(),0);
    Vec<Vec<Vec<Vec<float> > >* >        v_data_out_3D(numpyarrays.size(),0);
    Vec<Vec<Vec<Vec<Vec<float> > > >* >  v_data_out_4D(numpyarrays.size(),0);

    for(size_t i=0;i<numpyarrays.size();i++){
        Vec<int> shape = shapes.at(i);
        long numpyarray = numpyarrays.at(i);
        nentries = shape.at(0);
        TString add = adds.at(i);

        int ndims = shape.size() - 1;
        int ndims_shape=1;
        if(ndims)
            ndims_shape=ndims;

        std::vector<int> vecshape;

        size_t flattened_length = 1;
        for(size_t i = 1; i < shape.size(); i++) {
            vecshape.push_back(shape[i]);
            flattened_length *= shape[i];
        }
        shape=vecshape;
        v_shape.push_back(shape);
        if(storeflat){
            shape = std::vector<int>(1,flattened_length);
        }


        v_data_in.push_back(reinterpret_cast<float*>(numpyarray));
        v_ndims.push_back(ndims);
        v_flattened_length.push_back(flattened_length);

        if(ndims==0){
            v_data_out_0D.at(i)=create0DBranchVector(shape);
        }
        else if(ndims==1 || storeflat){
            v_data_out_1D.at(i)=create1DBranchVector(shape);
        }
        else if(ndims==2){
            v_data_out_2D.at(i)=create2DBranchVector(shape);
        }
        else if(ndims==3){
            v_data_out_3D.at(i)=create3DBranchVector(shape);
        }
        else if(ndims==4){
            v_data_out_4D.at(i)=create4DBranchVector(shape);
        }

    }



    Vec<Vec<int>* > v_shapep;
    for(size_t i=0;i<numpyarrays.size();i++){
        v_shapep.push_back(&v_shape.at(i));
        TString add = adds.at(i);
        if(add.IsAlnum())
            add="p_"+add;
        if(v_data_out_0D.at(i))
            t->Branch(add, v_data_out_0D.at(i));
        else if(v_data_out_1D.at(i)){
            t->Branch(add, &v_data_out_1D.at(i));
        }
        else if(v_data_out_2D.at(i)){
            t->Branch(add, &v_data_out_2D.at(i));
        }
        else if(v_data_out_3D.at(i)){
            t->Branch(add, &v_data_out_3D.at(i));
        }
        else if(v_data_out_4D.at(i)){
            t->Branch(add, &v_data_out_4D.at(i));
        }

    }


    if(storeflat){
        for(size_t i=0;i<numpyarrays.size();i++){
            if(v_ndims.at(i)){
                t->Branch((TString)adds.at(i)+"_shape",   v_shapep.at(i));
            }
        }
    }

    //vector<vector<... are not contiguous!

    for(size_t e = 0; e < nentries; e++) {

        for(size_t a=0;a<numpyarrays.size();a++){
            if(v_data_out_0D.at(a)){
                v_data_out_0D.at(a)[0]=v_data_in.at(a)[e*v_flattened_length.at(a)];
            }
            else if(v_data_out_1D.at(a)){
                for (size_t i = 0; i < v_data_out_1D.at(a)->size(); i++){
                    v_data_out_1D.at(a)->at(i)=v_data_in.at(a)[e*v_flattened_length.at(a)+i];
                }
            }
            else if(v_data_out_2D.at(a)){
                size_t globcont=0;
                for (size_t i = 0; i < v_data_out_2D.at(a)->size(); i++){
                    for (size_t j = 0; j < v_data_out_2D.at(a)->at(i).size(); j++){
                        v_data_out_2D.at(a)->at(i).at(j)=v_data_in.at(a)[e*v_flattened_length.at(a)+globcont];
                        globcont++;
                    }
                }
            }
            else if(v_data_out_3D.at(a)){
                size_t globcont=0;
                for (size_t i = 0; i < v_data_out_3D.at(a)->size(); i++){
                    for (size_t j = 0; j < v_data_out_3D.at(a)->at(i).size(); j++){
                        for (size_t k = 0; k < v_data_out_3D.at(a)->at(i).at(j).size(); k++){
                            v_data_out_3D.at(a)->at(i).at(j).at(k)=v_data_in.at(a)[e*v_flattened_length.at(a)+globcont];
                            globcont++;
                        }
                    }
                }
            }
            else if(v_data_out_4D.at(a)){
                size_t globcont=0;
                for (size_t i = 0; i < v_data_out_4D.at(a)->size(); i++){
                    for (size_t j = 0; j < v_data_out_4D.at(a)->at(i).size(); j++){
                        for (size_t k = 0; k < v_data_out_4D.at(a)->at(i).at(j).size(); k++){
                            for (size_t l = 0; l < v_data_out_4D.at(a)->at(i).at(j).at(k).size(); l++){
                                v_data_out_4D.at(a)->at(i).at(j).at(k).at(l)=v_data_in.at(a)[e*v_flattened_length.at(a)+globcont];
                                globcont++;
                            }
                        }
                    }
                }
            }
        }//arrays

        t->Fill();
    }


    t->Write();
    outfile->Close();
    delete outfile;


}








// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_storeTensor) {

    boost::python::numpy::initialize();
    __hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
    //anyway, it doesn't hurt, just leave this here
    def("store", &store);
}
