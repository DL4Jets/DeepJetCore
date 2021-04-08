/*
 * c_simpleArray.C
 *
 *  Created on: 16 Nov 2019
 *      Author: jkiesele
 *
 *  Simple reading and writing of numpy arrays using the simpleArray class
 *
 *  Only implemented for float32, int32 arrays
 *
 *   just a wrapper module
 */

#include "../interface/helper.h"
#include "../interface/simpleArray.h"
#include <cstdint>

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace djc;

using namespace p;
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(simpleArray_float32_set_overloads, simpleArray_float32::set, 2, 6);


BOOST_PYTHON_MODULE(c_simpleArray) {
    Py_Initialize();
    np::initialize();

    p::class_<simpleArray_float32 >("simpleArrayF")
        .def("readDtypeFromFile", &simpleArray_float32::readDtypeFromFile)

        .def(self==self)
        .def("dtypeI", &simpleArray_float32::dtypeI)

        .def("setName", &simpleArray_float32::setName)
        .def("name", &simpleArray_float32::name)
        .def("setFeatureNames", &simpleArray_float32::setFeatureNamesPy)
        .def("featureNames", &simpleArray_float32::featureNamesPy)

        .def("hasNanOrInf", &simpleArray_float32::hasNanOrInf)

        //explicit overloads necessary
        .def<void (simpleArray_float32::*)(const size_t i, float val)>("set", &simpleArray_float32::set)
        .def<void (simpleArray_float32::*)(const size_t i, const size_t j, float val)>("set", &simpleArray_float32::set)
        .def<void (simpleArray_float32::*)(const size_t i, const size_t j, const size_t k, float val)>("set", &simpleArray_float32::set)
        .def<void (simpleArray_float32::*)(const size_t i, const size_t j, const size_t k, const size_t l, float val)>("set", &simpleArray_float32::set)
        .def<void (simpleArray_float32::*)(const size_t i, const size_t j, const size_t k, const size_t l, const size_t m, float val)>("set", &simpleArray_float32::set)


        .def("readFromFile", &simpleArray_float32::readFromFile)
        .def("writeToFile", &simpleArray_float32::writeToFile)
        .def("assignFromNumpy", &simpleArray_float32::assignFromNumpy)
        .def("transferToNumpy", &simpleArray_float32::transferToNumpy)
        .def("createFromNumpy", &simpleArray_float32::createFromNumpy)
        .def("copyToNumpy", &simpleArray_float32::copyToNumpy)
        .def("isRagged", &simpleArray_float32::isRagged)
        .def("split", &simpleArray_float32::split)
        .def("getSlice", &simpleArray_float32::getSlice)
        .def<void (simpleArray_float32::*)(const simpleArray_float32&)>("append", &simpleArray_float32::append)
        .def("cout", &simpleArray_float32::cout)
        .def("size", &simpleArray_float32::isize)
        .def("shape", &simpleArray_float32::shapePy);
    ;
    p::class_<simpleArray_int32 >("simpleArrayI")
        .def("readDtypeFromFile", &simpleArray_int32::readDtypeFromFile)

        .def(self==self)

        .def("dtypeI", &simpleArray_int32::dtypeI)

        .def("setName", &simpleArray_int32::setName)
        .def("name", &simpleArray_int32::name)
        .def("setFeatureNames", &simpleArray_int32::setFeatureNamesPy)
        .def("featureNames", &simpleArray_int32::featureNamesPy)

        .def("hasNanOrInf", &simpleArray_int32::hasNanOrInf)

        .def<void (simpleArray_int32::*)(const size_t i, int val)>("set", &simpleArray_int32::set)
        .def<void (simpleArray_int32::*)(const size_t i, const size_t j, int val)>("set", &simpleArray_int32::set)
        .def<void (simpleArray_int32::*)(const size_t i, const size_t j, const size_t k, int val)>("set", &simpleArray_int32::set)
        .def<void (simpleArray_int32::*)(const size_t i, const size_t j, const size_t k, const size_t l, int val)>("set", &simpleArray_int32::set)
        .def<void (simpleArray_int32::*)(const size_t i, const size_t j, const size_t k, const size_t l, const size_t m, int val)>("set", &simpleArray_int32::set)


        .def("readFromFile", &simpleArray_int32::readFromFile)
        .def("writeToFile", &simpleArray_int32::writeToFile)
        .def("assignFromNumpy", &simpleArray_int32::assignFromNumpy)
        .def("transferToNumpy", &simpleArray_int32::transferToNumpy)
        .def("createFromNumpy", &simpleArray_int32::createFromNumpy)
        .def("copyToNumpy", &simpleArray_int32::copyToNumpy)
        .def("isRagged", &simpleArray_int32::isRagged)
        .def("split", &simpleArray_int32::split)
        .def("getSlice", &simpleArray_int32::getSlice)
        .def<void (simpleArray_int32::*)(const simpleArray_int32&)>("append", &simpleArray_int32::append) //just use the explicit one here
        .def("cout", &simpleArray_int32::cout)
        .def("size", &simpleArray_int32::isize)
        .def("shape", &simpleArray_int32::shapePy);
    ;

}

