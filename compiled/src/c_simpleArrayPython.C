/*
 * c_NumpyInterface.C
 *
 *  Created on: 6 Nov 2019
 *      Author: jkiesele
 *
 *  Simple reading and writing of numpy arrays using the simpleArray classes
 *  and quicklz. There is a lot of overhead w.r.t. standard numpy, but this implementation
 *  is faster, and mostly for occasional usage.
 *
 *  Only implemented for float32 arrays
 */

#include "../interface/helper.h"
#include "../interface/simpleArrayPython.h"


namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace djc;
/*
 * just a wrapper module
 */



BOOST_PYTHON_MODULE(c_simpleArrayPython) {
    Py_Initialize();
        np::initialize();

        p::class_<simpleArrayPython<float> >("simpleArrayPython")
                        .def("readFromFile", &simpleArrayPython<float>::readFromFile)
                        .def("writeToFile", &simpleArrayPython<float>::writeToFile)
                .def("assignFromNumpy", &simpleArrayPython<float>::assignFromNumpy)
                .def("transferToNumpy", &simpleArrayPython<float>::transferToNumpy)
                .def("isRagged", &simpleArrayPython<float>::isRagged)
                .def("split", &simpleArrayPython<float>::split)
                .def("append", &simpleArrayPython<float>::append);
            ;
}

