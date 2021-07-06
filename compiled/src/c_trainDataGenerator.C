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

#include <exception>

#include <iostream> //debug

namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace djc;


BOOST_PYTHON_MODULE(c_trainDataGenerator) {
    Py_Initialize();
    np::initialize();
    p::class_<trainDataGenerator >("trainDataGenerator")

            .def("setBatchSize", &trainDataGenerator::setBatchSize)

            .def("setFileList", &trainDataGenerator::setFileListPy)
            .def("shuffleFileList", &trainDataGenerator::shuffleFileList)

            .def("setBuffer", &trainDataGenerator::setBuffer)


            .def("setFileTimeout", &trainDataGenerator::setFileTimeout)
            .def("setSquaredElementsLimit", &trainDataGenerator::setSquaredElementsLimit)
            .def("setSkipTooLargeBatches", &trainDataGenerator::setSkipTooLargeBatches)

            .def("clear", &trainDataGenerator::clear)
            .def("getNBatches", &trainDataGenerator::getNBatches)

            .def("lastBatch", &trainDataGenerator::lastBatch)
            .def("isEmpty", &trainDataGenerator::isEmpty)

            .def("prepareNextEpoch", &trainDataGenerator::prepareNextEpoch)
            .def("getBatch", &trainDataGenerator::getBatch)

            .def("getNTotal", &trainDataGenerator::getNTotal)

            .def_readwrite("debuglevel", &trainDataGenerator::debuglevel);
        ;
}

