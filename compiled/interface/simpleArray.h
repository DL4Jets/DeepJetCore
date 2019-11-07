/*
 * simpleArray.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_

#include <vector>
#include <string>
#include <stdio.h>
#include "quicklzWrapper.h"
#include <cstring> //memcpy
#include "IO.h"
#include "version.h"

namespace djc{


template<class T>
class simpleArray {
public:

    simpleArray();
    simpleArray(std::vector<int> shape);
    simpleArray(FILE *& );
    ~simpleArray();

    simpleArray(const simpleArray<T>&);
    simpleArray<T>& operator=(const simpleArray<T>&);

    simpleArray(simpleArray<T> &&);
    simpleArray<T>& operator=(simpleArray<T> &&);

    void clear();

    //reshapes if possible, creates new else
    void setShape(std::vector<int> shape);

    T * data() const {
        return data_;
    }

    T * data() {
        return data_;
    }


    const std::vector<int>& shape() const {
        return shape_;
    }

    const size_t& size() const {
        return size_;
    }

    /*
     * returns the dimension of the first axis.
     * If second dimension is ragged, this will take it into
     * account.
     */
    size_t getFirstDimension()const;

    // row splits are indicated by a merged dimension with negative sign
    // always merge to previous dimension
    // e.g. A x B x C x D, where B is ragged would get shape
    // -A*B x C x D
    // Only one ragged dimension is supported!
    // Still to be implemented. All read/write functions already include this data
    //
    // const std::vector<int>& rowsplits() const {
    //     return rowsplits_;
    // }
    //
    // void setRowsplits(const std::vector<int>& rowsplits) {
    //     rowsplits_ = rowsplits;
    // }


    /////////// potentially dangerous operations for conversions, use with care ///////

    /*
     * Move data memory location to another object
     * This will reset the array. Read shapes, row splits etc. before
     * performing this operation!
     */
    T * disownData();

    /*
     * Object will not own the data. Merely useful for conversion
     * with immediate writing to file
     */
    void assignData(T *d){
        if(data_ && !assigned_)
            delete data_;
        data_=d;
        assigned_=true;
    }

    /*
     * Assigns a shape without checking it or creating a new data
     * array. Will recalculate total size
     */
    void assignShape(std::vector<int> s){
        shape_=s;
        size_ = sizeFromShape(s);
    }

    /*
     * Splits on first axis.
     * Returns the first part, leaves the second.
     * does memcopy for both pats now
     */
    simpleArray<T> split(size_t splitindex);

    /*
     * appends along first axis
     * Cann append to an empty array (same as copy)
     */
    void append(const simpleArray<T>& a);

    /* file IO here
     * format: non compressed header (already writing rowsplits!):
     * size, shape.size(), [shape], rowsplits.size(), [rowsplits], compr: [data]
     *
     */
    void addToFile(FILE *& ofile) const;

    void readFromFile(FILE *& ifile);

    /*
     * Does not work (yet) with ragged arrays!
     * Will just produce garbage!
     */

    T & at(size_t i);
    const T & at(size_t i)const;
    T & at(size_t i, size_t j);
    const T & at(size_t i, size_t j)const;
    T & at(size_t i, size_t j, size_t k);
    const T & at(size_t i, size_t j, size_t k)const;
    T & at(size_t i, size_t j, size_t k, size_t l);
    const T & at(size_t i, size_t j, size_t k, size_t l)const;
    T & at(size_t i, size_t j, size_t k, size_t l, size_t m);
    const T & at(size_t i, size_t j, size_t k, size_t l, size_t m)const;
    T & at(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n);
    const T & at(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)const;

private:
    void copyFrom(const simpleArray<T>& a);
    void moveFrom(simpleArray<T> && a);
    int sizeFromShape(std::vector<int> shape) const;
    void checkShape(size_t ndims)const;
    void checkSize(size_t idx)const;

    T * data_;
    std::vector<int> shape_;
    std::vector<int> rowsplits_;
    size_t size_;
    bool assigned_;
};

template<class T>
simpleArray<T>::simpleArray() :
        data_(0), size_(0),assigned_(false) {

}

template<class T>
simpleArray<T>::simpleArray(std::vector<int> shape) :
        data_(0), size_(0),assigned_(false) {
    shape_ = shape;
    data_ = new T[sizeFromShape(shape_)];
    size_ = sizeFromShape(shape_);
}

template<class T>
simpleArray<T>::simpleArray(FILE *& ifile):simpleArray<T>(){
    readFromFile(ifile);
    assigned_=false;
}

template<class T>
simpleArray<T>::~simpleArray() {
    clear();
}

template<class T>
simpleArray<T>::simpleArray(const simpleArray<T>& a) :
        simpleArray<T>() {
    copyFrom(a);
}

template<class T>
simpleArray<T>& simpleArray<T>::operator=(const simpleArray<T>& a) {
    copyFrom(a);
    return *this;
}

template<class T>
simpleArray<T>::simpleArray(simpleArray<T> && a) :
        simpleArray<T>() {
    if (&a == this)
        return;
    if (data_&& !assigned_)
        delete data_;
    data_ = a.data_;
    a.data_ = 0;
    assigned_ = a.assigned_;
    size_ = a.size_;
    a.size_ = 0;
    shape_ = std::move(a.shape_);
    rowsplits_ = std::move(a.rowsplits_);
}

template<class T>
simpleArray<T>& simpleArray<T>::operator=(simpleArray<T> && a) {
    if (&a == this)
        return *this;
    if (data_ && !assigned_)
        delete data_;
    data_ = a.data_;
    a.data_ = 0;
    size_ = a.size_;
    assigned_ = a.assigned_;
    a.size_ = 0;
    shape_ = std::move(a.shape_);
    rowsplits_ = std::move(a.rowsplits_);
    return *this;
}

template<class T>
void simpleArray<T>::clear() {
    if (data_&& !assigned_)
        delete data_;
    data_ = 0;
    shape_.clear();
    rowsplits_.clear();
    size_ = 0;
}

template<class T>
void simpleArray<T>::setShape(std::vector<int> shape) {
    int size = sizeFromShape(shape);
    if (size != size_) {
        *this = simpleArray<T>(shape);
    } else if (size == size_) {
        shape_ = shape;
    }
}
template<class T>
size_t simpleArray<T>::getFirstDimension()const{
    if(!size_ || !shape_.size())
        return 0;
    if(shape_.at(0)>0)
        return shape_.at(0);
    else
        return rowsplits_.size();
}

template<class T>
T * simpleArray<T>::disownData() {
    T * dp = data_;
    data_ = 0;
    clear();
    return dp;
}

/*
 * Splits on first axis.
 * Returns the first part, leaves the second
 */
template<class T>
simpleArray<T> simpleArray<T>::split(size_t splitindex) {
    simpleArray<T> out;
    if (!shape_.size() || splitindex > shape_.at(0)) {
        throw std::runtime_error(
                "simpleArray<T>::split: splitindex > shape_[0]");
    }
    if (rowsplits_.size()) {
        throw std::runtime_error("simpleArray<T>::split: TBI for row splits");
        //check row split dimension, check if split possible and split accordingly
    }
    //get split point for data
    size_t splitpoint = splitindex;
    for (size_t i = 1; i < shape_.size(); i++)
        splitpoint *= std::abs(shape_.at(i));
    size_t remaining = size_ - splitpoint;

    T * odata = new T[splitpoint];
    T * rdata = new T[remaining];

    memcpy(odata, data_, splitpoint * sizeof(T));
    memcpy(rdata, data_ + splitpoint, remaining * sizeof(T));
    if(!assigned_)
        delete data_;
    out.data_ = odata;
    data_ = rdata;
    out.shape_ = shape_;
    out.shape_.at(0) = splitindex;
    shape_.at(0) = shape_.at(0) - splitindex;
    out.size_ = sizeFromShape(out.shape_);
    size_ = sizeFromShape(shape_);
    return out;
}

/*
 * Merges along first axis
 */
template<class T>
void simpleArray<T>::append(const simpleArray<T>& a) {

    if (!data_ && size_ == 0) {
        *this = a;
        return;
    }
    if (shape_.size() != a.shape_.size())
        throw std::out_of_range(
                "simpleArray<T>::append: shape dimensions don't match");
    std::vector<int> targetshape;
    if (shape_.size() > 1 && a.shape_.size() > 1) {
        std::vector<int> highshape = std::vector<int>(shape_.begin() + 1,
                shape_.end());
        std::vector<int> ahighshape = std::vector<int>(a.shape_.begin() + 1,
                a.shape_.end());
        if (highshape != ahighshape) {
            throw std::out_of_range(
                    "simpleArray<T>::append: all shapes but first axis must match");
        }
        targetshape.push_back(shape_.at(0) + a.shape_.at(0));
        targetshape.insert(targetshape.end(), highshape.begin(),
                highshape.end());
    } else {
        targetshape.push_back(shape_.at(0) + a.shape_.at(0));
    }

    if (rowsplits_.size() || a.rowsplits_.size())
        throw std::runtime_error("simpleArray<T>::append: TBI for row splits");

    T * ndata = new T[size_ + a.size_];
    memcpy(ndata, data_, size_ * sizeof(T));
    memcpy(ndata + size_, a.data_, a.size_ * sizeof(T));
    if(!assigned_)
        delete data_;
    data_ = ndata;
    size_ = size_ + a.size_;
    shape_ = targetshape;

}

template<class T>
void simpleArray<T>::addToFile(FILE *& ofile) const {

    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);
    io::writeToFile(&size_, ofile);
    size_t ssize = shape_.size();
    io::writeToFile(&ssize, ofile);
    io::writeToFile(&shape_[0], ofile, shape_.size());
    size_t rssize = rowsplits_.size();
    io::writeToFile(&rssize,  ofile);
    io::writeToFile(&rowsplits_[0], ofile, rowsplits_.size());

    quicklz<T> qlz;
    qlz.writeCompressed(data_, size_, ofile);
}

template<class T>
void simpleArray<T>::readFromFile(FILE *& ifile) {
    clear();

    float version = 0;
    io::readFromFile(&version, ifile);
    if(version != DJCDATAVERSION)
        throw std::runtime_error("simpleArray<T>::readFromFile: wrong format version");

    io::readFromFile(&size_, ifile);

    size_t shapesize = 0;
    io::readFromFile(&shapesize, ifile);
    shape_ = std::vector<int>(shapesize, 0);
    io::readFromFile(&shape_[0], ifile, shapesize);

    size_t rssize = 0;
    io::readFromFile(&rssize, ifile);
    rowsplits_ = std::vector<int>(rssize, 0);
    io::readFromFile(&rowsplits_[0], ifile, rssize);


    data_ = new T[size_];
    quicklz<T> qlz;
    size_t nread = qlz.readAll(ifile, data_);
    if (nread != size_)
        throw std::runtime_error(
                "simpleArray<T>::readFromFile: expected and observed length don't match");

}
template<class T>
void simpleArray<T>::copyFrom(const simpleArray<T>& a) {

    if (&a == this) {
        return;
    }
    if (data_&& !assigned_)
        delete data_;
    data_ = new T[a.size_];
    memcpy(data_, a.data_, a.size_ * sizeof(T));

    size_ = a.size_;
    shape_ = a.shape_;
    rowsplits_ = a.rowsplits_;
    assigned_=false;
}

template<class T>
int simpleArray<T>::sizeFromShape(std::vector<int> shape) const {
    int size = 1;
    for (const auto s : shape)
        size *= std::abs(s);
    return size;
}
template<class T>
void simpleArray<T>::checkShape(size_t ndims)const{
    if(rowsplits_.size()){
        //TBI
    }
    if(ndims != shape_.size()){
        throw std::out_of_range("simpleArray<T>::checkShape: shape does not match dimensions accessed");
    }
}

template<class T>
void simpleArray<T>::checkSize(size_t idx)const{
    if(idx >= size_)
        throw std::out_of_range("simpleArray<T>::checkSize: index out of range");
}



template<class T>
T & simpleArray<T>::at(size_t i){
    checkShape(1);
    checkSize(i);
    return data_[i];
}

template<class T>
const T & simpleArray<T>::at(size_t i)const{
    checkShape(1);
    checkSize(i);
    return data_[i];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j){
    checkShape(2);
    size_t flat = j + shape_.at(1)*i;
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j)const{
    checkShape(2);
    size_t flat = j + shape_.at(1)*i;
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k){
    checkShape(3);
    size_t flat = k + shape_.at(2)*(j + shape_.at(1)*i);
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k)const{
    checkShape(3);
    size_t flat = k + shape_.at(2)*(j + shape_.at(1)*i);
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l){
    checkShape(4);
    size_t flat = l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i));
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l)const{
    checkShape(4);
    size_t flat = l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i));
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m){
    checkShape(5);
    size_t flat = m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i)));
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m)const{
    checkShape(5);
    size_t flat = m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i)));
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){
    checkShape(6);
    size_t flat = n + shape_.at(5)*(m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i))));
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)const{
    checkShape(6);
    size_t flat = n + shape_.at(5)*(m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i))));
    checkSize(flat);
    return data_[flat];
}

}

#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_ */
