/*
 * c_TrainDataInferface.C
 *
 *  Created on: 6 Nov 2019
 *      Author: jkiesele
 */


#include "../interface/trainData.h"

#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
#include <boost/python/exception_translator.hpp>
#include <exception>

#include <iostream> //debug

namespace p = boost::python;
namespace np = boost::python::numpy;

/*
 * write out a cpp traindata object from a python traindata object using
 * three lists: x, y, w
 * make choice for float here
 */

float * extractNumpyListElement(p::list x, int i, std::vector<int>& shape){
    shape.clear();
    np::ndarray ndarr = p::extract<np::ndarray>(x[i]);
    float * data = (float*)(void*) ndarr.get_data();
    int ndim = ndarr.get_nd();
    for(int s=0;s<ndim;s++)
        shape.push_back(ndarr.shape(s));
    return data;
}

void writeToFile(p::list x, p::list y, p::list w, std::string filename){

    djc::trainData<float> td;

    int nx = p::extract<int>(x.attr("__len__")());
    int ny = p::extract<int>(x.attr("__len__")());
    int nw = p::extract<int>(x.attr("__len__")());

    std::vector<int> shape;
    for(int i=0;i<nx;i++){
        size_t idx = td.addFeatureArray({0});
        float * data = extractNumpyListElement(x,i,shape);
        td.featureArray(idx).assignData(data);
        td.featureArray(idx).assignShape(shape);
    }

    for(int i=0;i<ny;i++){
        size_t idx = td.addTruthArray({0});
        float * data = extractNumpyListElement(x,i,shape);
        td.truthArray(idx).assignData(data);
        td.truthArray(idx).assignShape(shape);
    }

    for(int i=0;i<nw;i++){
        size_t idx = td.addWeightArray({0});
        float * data = extractNumpyListElement(x,i,shape);
        td.weightArray(idx).assignData(data);
        td.weightArray(idx).assignShape(shape);
    }

    td.writeToFile(filename);


}


/*
 * Read a cpp traindata Object to its python counterpart.
 * Only for debugging/plotting purposes
 * return list of list of numpy arrays
 */


np::ndarray simpleArrayToNumpy( djc::simpleArray<float>& ifarr);

p::list readFromFile(std::string filename){
    p::list out;

    djc::trainData<float> td;
    td.readFromFile(filename);

    p::list x, y, w;

    for(size_t i=0;i<td.nFeatureArrays();i++)
        x.append(simpleArrayToNumpy(td.featureArray(i)));

    for(size_t i=0;i<td.nFeatureArrays();i++)
        y.append(simpleArrayToNumpy(td.truthArray(i)));

    for(size_t i=0;i<td.nFeatureArrays();i++)
        w.append(simpleArrayToNumpy(td.weightArray(i)));

    out.append(x);
    out.append(y);
    out.append(w);

    std::cout << "done" <<std::endl;

    return out;
}


np::ndarray simpleArrayToNumpy( djc::simpleArray<float>& ifarr){

    auto size = ifarr.size();
    auto shape =  ifarr.shape();
    for(const auto& s:shape){
        if(s<0)
            throw std::runtime_error("simpleArrayToNumpy: no conversion from ragged simpleArrys possible");
    }

    p::list pshape;
    for(const auto& s:shape)
        pshape.append(s);

    p::tuple tshape(pshape);//not working

    np::ndarray nparr = np::from_data((void*)ifarr.disownData(),
            np::dtype::get_builtin<float>(),
            p::make_tuple(size), p::make_tuple(sizeof(float)), p::object() );

    nparr = nparr.reshape(tshape);
    return nparr;

}

BOOST_PYTHON_MODULE(c_trainDataInterface) {
    Py_Initialize();
    np::initialize();
    def("writeToFile", &writeToFile);
    def("readFromFile", &readFromFile);
}



