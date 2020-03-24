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

            .def("setBatchSize", &trainDataGenerator<float>::setBatchSize)

            .def("setFileList", &trainDataGenerator<float>::setFileListP)
            .def("shuffleFilelist", &trainDataGenerator<float>::shuffleFilelist)

            .def("setBuffer", &trainDataGenerator<float>::setBuffer)


            .def("setFileTimeout", &trainDataGenerator<float>::setFileTimeout)
            .def("setSquaredElementsLimit", &trainDataGenerator<float>::setSquaredElementsLimit)
            .def("setSkipTooLargeBatches", &trainDataGenerator<float>::setSkipTooLargeBatches)

            .def("clear", &trainDataGenerator<float>::clear)
            .def("getNBatches", &trainDataGenerator<float>::getNBatches)

            .def("lastBatch", &trainDataGenerator<float>::lastBatch)
            .def("isEmpty", &trainDataGenerator<float>::isEmpty)

            .def("prepareNextEpoch", &trainDataGenerator<float>::prepareNextEpoch)
            .def("getBatch", &trainDataGenerator<float>::getBatch)

            .def("getNTotal", &trainDataGenerator<float>::getNTotal)

            .def_readwrite("debuglevel", &trainDataGenerator<float>::debuglevel);
        ;
}

