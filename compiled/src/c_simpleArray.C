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

//switches on python / numpy interfaces in the templates included below
#define DJC_DATASTRUCTURE_PYTHON_BINDINGS
#include "../interface/helper.h"
#include "../interface/simpleArray.h"
#include <cstdint>

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace djc;


BOOST_PYTHON_MODULE(c_simpleArray) {
    Py_Initialize();
    np::initialize();

    using namespace p;
    p::class_<simpleArray_float32 >("simpleArrayF")
            .def("readDtypeFromFile", &simpleArray_float32::readDtypeFromFile)

            .def(self==self)
            .def("dtypeI", &simpleArray_float32::dtypeI)

            .def("setName", &simpleArray_float32::setName)
            .def("name", &simpleArray_float32::name)
            .def("setFeatureNames", &simpleArray_float32::setFeatureNamesPy)
            .def("featureNames", &simpleArray_float32::featureNamesPy)

       .def("readFromFile", &simpleArray_float32::readFromFile)
       .def("writeToFile", &simpleArray_float32::writeToFile)
       .def("assignFromNumpy", &simpleArray_float32::assignFromNumpy)
       .def("transferToNumpy", &simpleArray_float32::transferToNumpy)
       .def("createFromNumpy", &simpleArray_float32::createFromNumpy)
       .def("copyToNumpy", &simpleArray_float32::copyToNumpy)
       .def("isRagged", &simpleArray_float32::isRagged)
       .def("split", &simpleArray_float32::split)
       .def("getSlice", &simpleArray_float32::getSlice)
       .def<void (simpleArray_float32::*)(const simpleArray_float32&)>("append", &simpleArray_float32::append) //just use the explicit one here
       .def("cout", &simpleArray_float32::cout)
       .def("size", &simpleArray_float32::isize);
    ;
    p::class_<simpleArray_int32 >("simpleArrayI")
                    .def("readDtypeFromFile", &simpleArray_int32::readDtypeFromFile)

                    .def(self==self)

                    .def("dtypeI", &simpleArray_int32::dtypeI)

                    .def("setName", &simpleArray_int32::setName)
                    .def("name", &simpleArray_int32::name)
                    .def("setFeatureNames", &simpleArray_int32::setFeatureNamesPy)
                    .def("featureNames", &simpleArray_int32::featureNamesPy)

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
       .def("size", &simpleArray_int32::isize);
    ;

}

