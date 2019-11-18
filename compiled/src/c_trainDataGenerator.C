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

#define DJC_DATASTRUCTURE_PYTHON_BINDINGS

#include "../interface/trainDataGenerator.h"

#include <exception>

#include <iostream> //debug

namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace djc;


BOOST_PYTHON_MODULE(c_trainDataGenerator) {
    Py_Initialize();
    np::initialize();
    p::class_<trainDataGenerator<float> >("trainDataGenerator")

            .def("setFileList", &trainDataGenerator<float>::setFileListP)
            .def("setBatchSize", &trainDataGenerator<float>::setBatchSize)
            .def("shuffleFilelist", &trainDataGenerator<float>::shuffleFilelist)

            .def("setFileTimeout", &trainDataGenerator<float>::setFileTimeout)

            .def("getNBatches", &trainDataGenerator<float>::getNBatches)

            .def("lastBatch", &trainDataGenerator<float>::lastBatch)
            .def("prepareNextEpoch", &trainDataGenerator<float>::prepareNextEpoch)
            .def("getBatch", &trainDataGenerator<float>::getBatch)


            .def_readwrite("debug", &trainDataGenerator<float>::debug);
        ;
}

