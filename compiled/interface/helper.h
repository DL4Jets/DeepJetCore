/*
 * helper.h
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_HELPER_H_
#define DEEPJET_MODULES_INTERFACE_HELPER_H_


#include <boost/python.hpp>
#include "boost/python/numpy.hpp"

#include <dirent.h>
#include <stdlib.h>
#include "TString.h"
#include "TObject.h"
#include "TString.h"

TString prependXRootD(const TString& path);

bool isApprox(const float& a , const float& b, float eps=0.001);

float deltaPhi(const float& phi1, const float& phi2);

void checkTObject(const TObject * o, TString msg);


/**
 * transfers ownership of the data to numpy array if no copy.
 * size given it nobjects, not in bytes
 */
template<class T>
boost::python::numpy::ndarray STLToNumpy(const T * data, const std::vector<int>& shape, const size_t& size, bool copy=true);



//////// template implementations

namespace _hidden{
inline void destroyManagerCObject(PyObject* self) {
    auto * b = reinterpret_cast<float*>( PyCapsule_GetPointer(self, NULL) );
    delete [] b;
}
}


template<class T>
boost::python::numpy::ndarray STLToNumpy(const T * data, const std::vector<int>& shape, const size_t& size, bool copy){

    namespace p = boost::python;
    namespace np = boost::python::numpy;

    if(size>0){
        p::list pshape;
        size_t sizecheck = 1;
        for(size_t i=0;i<shape.size();i++){
            pshape.append(shape.at(i));
            sizecheck *= shape.at(i);
        }
        if(sizecheck != size)
            throw std::out_of_range("STLToNumpy: shape and size don't match");

        p::tuple tshape(pshape);

        T * data_ptr = (T *)(void *)data;
        if(copy){
            data_ptr = new T[size];
            memcpy(data_ptr,data,size*sizeof(T));
        }

        PyObject *capsule = ::PyCapsule_New((void *)data_ptr, NULL, (PyCapsule_Destructor)&_hidden::destroyManagerCObject);
        boost::python::handle<> h_capsule{capsule};
        boost::python::object owner_capsule{h_capsule};

        np::ndarray dataarr = np::from_data((void*)data_ptr,
                np::dtype::get_builtin<T>(),
                p::make_tuple(size), p::make_tuple(sizeof(T)), owner_capsule );
        dataarr = dataarr.reshape(tshape);

        return dataarr;
    }
    else{
        return np::empty(p::make_tuple(0), np::dtype::get_builtin<T>());;
    }
}


#endif /* DEEPJET_MODULES_INTERFACE_HELPER_H_ */
