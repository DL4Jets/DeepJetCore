/*
 * c_dataGenerator.C
 *
 *  Created on: 7 Nov 2019
 *      Author: jkiesele
 */



#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
#include <boost/python/exception_translator.hpp>

#include "../interface/trainDataGenerator.h"
#include "../interface/helper.h"
#include "../interface/pythonToSTL.h"

#include <exception>

#include <iostream> //debug

namespace p = boost::python;
namespace np = boost::python::numpy;


class numpyGenerator : public djc::trainDataGenerator<float>{
public:
    numpyGenerator():djc::trainDataGenerator<float>(){
    }

    void setFileList(p::list files){
        djc::trainDataGenerator<float>::setFileList(toSTLVector<std::string>(files));
    }

    p::list getBatch();

};


p::list numpyGenerator::getBatch(){

    p::list out;

    auto td = djc::trainDataGenerator<float>::getBatch();

    p::list x, y, w;

    for(size_t i=0;i<td.nFeatureArrays();i++)
        x.append(simpleArrayToNumpy(td.featureArray(i)));

    for(size_t i=0;i<td.nTruthArrays();i++)
        y.append(simpleArrayToNumpy(td.truthArray(i)));

    for(size_t i=0;i<td.nWeightArrays();i++)
        w.append(simpleArrayToNumpy(td.weightArray(i)));

    out.append(x);
    out.append(y);
    out.append(w);

    return out;
}


BOOST_PYTHON_MODULE(c_dataGenerator) {
    Py_Initialize();
    np::initialize();
    p::class_<numpyGenerator>("numpyGenerator")
            .def("setFileList", &numpyGenerator::setFileList)
            .def("setBatchSize", &numpyGenerator::setBatchSize)
            .def("getNTotal", &numpyGenerator::getNTotal)
            .def("setFileTimeout", &numpyGenerator::setFileTimeout)
            .def("getNBatches", &numpyGenerator::getNBatches)
            .def("lastBatch", &numpyGenerator::lastBatch)
            .def("prepareNextEpoch", &numpyGenerator::prepareNextEpoch)
            .def("getBatch", &numpyGenerator::getBatch)
            .def_readwrite("debug", &numpyGenerator::debug);
        ;
}

