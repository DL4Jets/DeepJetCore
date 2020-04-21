/*
 * c_simpleArray.C
 *
 *  Created on: 16 Nov 2019
 *      Author: jkiesele
 *
 *  Simple reading and writing of numpy arrays using the simpleArray class
 *
 *  Only implemented for float32 arrays
 *
 *   just a wrapper module
 */

//switches on python / numpy interfaces in the templates included below
#define DJC_DATASTRUCTURE_PYTHON_BINDINGS
#include "../interface/helper.h"
#include "../interface/simpleArray.h"


namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace djc;


BOOST_PYTHON_MODULE(c_simpleArray) {
    Py_Initialize();
    np::initialize();
    p::class_<simpleArray<float> >("simpleArray")
       .def("readFromFile", &simpleArray<float>::readFromFile)
       .def("writeToFile", &simpleArray<float>::writeToFile)
       .def("assignFromNumpy", &simpleArray<float>::assignFromNumpy)
       .def("transferToNumpy", &simpleArray<float>::transferToNumpy)
       .def("createFromNumpy", &simpleArray<float>::createFromNumpy)
       .def("copyToNumpy", &simpleArray<float>::copyToNumpy)
       .def("isRagged", &simpleArray<float>::isRagged)
       .def("split", &simpleArray<float>::split)
       .def("getSlice", &simpleArray<float>::getSlice)
       .def("append", &simpleArray<float>::append)
       .def("cout", &simpleArray<float>::cout)
       .def("size", &simpleArray<float>::isize);
    ;
}

