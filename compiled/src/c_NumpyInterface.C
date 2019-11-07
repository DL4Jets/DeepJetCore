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

#include "../interface/simpleArray.h"
#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include <exception>


namespace p = boost::python;
namespace np = boost::python::numpy;


np::ndarray readFromFile(std::string filename, bool shapesOnly){



    FILE *ifile = fopen("testfile.djcd", "rb");
    djc::simpleArray<float> ifarr(ifile);
    fclose(ifile);

    auto size = ifarr.size();
    auto shape =  ifarr.shape();

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

BOOST_PYTHON_MODULE(c_NumpyInterface) {
    Py_Initialize();
    boost::python::numpy::initialize();
    def("readFromFile", &readFromFile);
}

