/*
 * c_TrainDataInferface.C
 *
 *  Created on: 6 Nov 2019
 *      Author: jkiesele
 */


#include "../interface/trainData.h"
#include "../interface/helper.h"

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
 *
 * On longer term make this a real class wrapper. So far ok, still needs
 * rework for ragged anyway
 *
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
    int ny = p::extract<int>(y.attr("__len__")());
    int nw = p::extract<int>(w.attr("__len__")());

    std::vector<int> shape;
    for(int i=0;i<nx;i++){
        size_t idx = td.addFeatureArray({0});
        float * data = extractNumpyListElement(x,i,shape);
        td.featureArray(idx).assignData(data);
        td.featureArray(idx).assignShape(shape);
    }

    for(int i=0;i<ny;i++){
        size_t idx = td.addTruthArray({0});
        float * data = extractNumpyListElement(y,i,shape);
        td.truthArray(idx).assignData(data);
        td.truthArray(idx).assignShape(shape);
    }

    for(int i=0;i<nw;i++){
        size_t idx = td.addWeightArray({0});
        float * data = extractNumpyListElement(w,i,shape);
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

p::list nestedToList(const std::vector<std::vector<int> >& v){
    p::list out;
    for(const auto& vv:v){
        p::list tmp;
        for(const auto& vvv:vv)
            tmp.append(vvv);
        out.append(tmp);
    }
    return out;
}

p::list readShapesFromFile(std::string filename){
    p::list out;
    djc::trainData<float> td;
    std::vector<std::vector<int> > fs,ts,ws;
    td.readShapesFromFile(filename,fs,ts,ws);
    out.append(nestedToList(fs));
    out.append(nestedToList(ts));
    out.append(nestedToList(ws));
    return out;
}



BOOST_PYTHON_MODULE(c_trainDataInterface) {
    Py_Initialize();
    np::initialize();
    def("writeToFile", &writeToFile);
    def("readFromFile", &readFromFile);
    def("readShapesFromFile", &readShapesFromFile);
}



