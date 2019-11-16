/*
 * simpleArrayPython.h
 *
 *  Created on: 16 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAYPYTHON_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAYPYTHON_H_


#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include <exception>

#include "simpleArray.h"

#include <iostream>
/*
 * slim simpleArray <-> python, boost numpy interface wrapper
 */
namespace djc{

template<class T>
class simpleArrayPython : public simpleArray<T> {
public:
    //pass through
    simpleArrayPython():simpleArray<T>(){}
    simpleArrayPython(std::vector<int> shape,const std::vector<size_t>& rowsplits = {}):simpleArray<T>(shape,rowsplits) {}
    simpleArrayPython(FILE *& f):simpleArray<T>(f){}

    simpleArrayPython(simpleArrayPython<T> && rhs):simpleArray<T>(std::move(rhs)){}
    simpleArrayPython<T>& operator=(simpleArrayPython<T> && rhs){simpleArray<T>::operator =(rhs);return *this;}

    simpleArrayPython(const simpleArrayPython<T>& rhs): simpleArray<T>::simpleArray(rhs){}
    simpleArrayPython<T>& operator=(const simpleArrayPython<T>& rhs){simpleArray<T>::operator =(rhs);return *this;}

    //(destructor automatically inherited)

    //convert on the fly
    simpleArrayPython(const simpleArray<T>& rhs): simpleArray<T>::simpleArray(rhs){}
    simpleArrayPython<T>& operator=(const simpleArray<T>& rhs){simpleArray<T>::operator =(rhs);return *this;}

    simpleArrayPython<T> split(size_t splitindex){return simpleArray<T>::split(splitindex);}


    ///new stuff

    //does not transfer data ownership! only for quick writing etc.
    void assignFromNumpy(const boost::python::numpy::ndarray& ndarr,
            const boost::python::numpy::ndarray& rowsplits=boost::python::numpy::empty(
                    boost::python::make_tuple(0), boost::python::numpy::dtype::get_builtin<size_t>()));

    //transfers data ownership and cleans simpleArray instance
    boost::python::tuple transferToNumpy();


private:

    void checkArray(const boost::python::numpy::ndarray& ndarr,
            boost::python::numpy::dtype dt=boost::python::numpy::dtype::get_builtin<T>())const;
    //void addToFile(FILE *& ofile) const{}

    //void readFromFile(FILE *& ifile){}


};

template<class T>
void simpleArrayPython<T>::checkArray(const boost::python::numpy::ndarray& ndarr,
        boost::python::numpy::dtype dt)const{
    namespace p = boost::python;
    namespace np = boost::python::numpy;

    if(ndarr.get_dtype() != dt){
        throw std::runtime_error("c_trainDataInterface.extractNumpyListElement: at least one array does not have right type.");
    }
    auto flags = ndarr.get_flags();
    if(!(flags & np::ndarray::CARRAY) || !(flags & np::ndarray::C_CONTIGUOUS)){
        throw std::runtime_error("simpleArrayPython<T>::assignFromNumpy: at least one array is not C contiguous, please pass as numpy.ascontiguousarray(a, dtype='float32')");
    }
}

template<class T>
void simpleArrayPython<T>::assignFromNumpy(const boost::python::numpy::ndarray& ndarr,
        const boost::python::numpy::ndarray& rowsplits){
    namespace p = boost::python;
    namespace np = boost::python::numpy;

    this->clear();
    checkArray(ndarr, np::dtype::get_builtin<T>());

    this->data_ = (float*)(void*) ndarr.get_data();

    int ndim = ndarr.get_nd();
    std::vector<int> shape;
    for(int s=0;s<ndim;s++)
        shape.push_back(ndarr.shape(s));

    //check row splits
    if(len(rowsplits)){
        checkArray(rowsplits, np::dtype::get_builtin<size_t>());
        this->rowsplits_.resize(len(rowsplits));
        memcpy(&(this->rowsplits_.at(0)),(size_t*)(void*) rowsplits.get_data(), this->rowsplits_.size() * sizeof(size_t));
        shape.insert(shape.begin(),len(rowsplits));
        this->shape_ = shape;
        this->shape_ = this->shapeFromRowsplits();
    }
    else{
        this->shape_ = shape;
    }
    this->size_ = this->sizeFromShape(this->shape_);
    this->assigned_=true;
    std::cout << "converted " <<std::endl;
    this->cout();
}


inline void destroyManagerCObject(PyObject* self) {
    auto * b = reinterpret_cast<float*>( PyCapsule_GetPointer(self, NULL) );
    delete [] b;
}
//transfers data ownership and cleans simpleArray instance
template<class T>
boost::python::tuple simpleArrayPython<T>::transferToNumpy(){

    namespace p = boost::python;
    namespace np = boost::python::numpy;


    auto size = this->size();
    auto shape =  this->shape();

    std::cout << "size " << size << std::endl;

    p::list pshape;
    for(size_t i=0;i<this->shape().size();i++){
        if(this->isRagged() && i<1)continue;
        pshape.append(this->shape().at(i));
    }

    p::tuple tshape(pshape);

    T * data_ptr = this->disownData();
    //ifarr invalid from here on!

    PyObject *capsule = ::PyCapsule_New((void *)data_ptr, NULL, (PyCapsule_Destructor)&destroyManagerCObject);
    boost::python::handle<> h_capsule{capsule};
    boost::python::object owner_capsule{h_capsule};

    np::ndarray dataarr = np::from_data((void*)data_ptr,
            np::dtype::get_builtin<T>(),
            p::make_tuple(size), p::make_tuple(sizeof(T)), owner_capsule );

    dataarr = dataarr.reshape(tshape);

    // row splits
    // uintp is size_t in numpy

    size_t * rs = 0;
    if(this->isRagged()){
        rs = new size_t [this->rowsplits_.size()];
        memcpy(rs, &this->rowsplits_.at(0), this->rowsplits_.size() * sizeof(size_t));

        PyObject *capsule2 = ::PyCapsule_New((void *)rs, NULL, (PyCapsule_Destructor)&destroyManagerCObject);
        boost::python::handle<> h_capsule2{capsule2};
        boost::python::object owner_capsule2{h_capsule2};

        np::ndarray rowsplits = np::from_data((void*)rs,
                np::dtype::get_builtin<size_t>(),
                p::make_tuple(this->rowsplits_.size()), p::make_tuple(sizeof(size_t)), owner_capsule2 );

        this->clear();//reset all
        return p::make_tuple(dataarr,rowsplits);
    }
    else{
        np::ndarray rowsplits = np::empty(p::make_tuple(0), np::dtype::get_builtin<size_t>());
        this->clear();//reset all
        return p::make_tuple(dataarr,rowsplits);
    }

}



}//namespace





#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAYPYTHON_H_ */
