/*
 * simpleArray.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_

#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include "helper.h"
#endif

#include <cmath>
#include <vector>
#include <string>
#include <stdio.h>
#include "quicklzWrapper.h"
#include <cstring> //memcpy
#include "IO.h"
#include "version.h"
#include <iostream>
#include <cstdint>
#include <sstream>

namespace djc{


template<class T>
class simpleArray {
public:

    simpleArray();
    // row splits are indicated by a merged dimension with negative sign
    // e.g. A x B x C x D, where B is ragged would get shape
    // A x -nElementsTotal x C x D
    // ROW SPLITS START WITH 0 and end with the total number of elements along that dimension
    // therefore, the rosplits vector size is one more than the first dimension
    //
    // Only ONLY DIMENSION 1 AS RAGGED DIMENSION is supported, first dimension MUST NOT be ragged.
    //

    simpleArray(std::vector<int> shape,const std::vector<int64_t>& rowsplits = {});
    simpleArray(FILE *& );
    ~simpleArray();

    simpleArray(const simpleArray<T>&);
    simpleArray<T>& operator=(const simpleArray<T>&);

    simpleArray(simpleArray<T> &&);
    simpleArray<T>& operator=(simpleArray<T> &&);

    void clear();

    //reshapes if possible, creates new else
    void setShape(std::vector<int> shape,const std::vector<int64_t>& rowsplits = {});

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

    bool isRagged()const{
        return rowsplits_.size()>0;
    }

    /*
     * returns the dimension of the first axis.
     * If second dimension is ragged, this will take it into
     * account.
     */
    size_t getFirstDimension()const;

    const std::vector<int64_t>& rowsplits() const {
        return rowsplits_;
    }

    /////////// potentially dangerous operations for conversions, use with care ///////

    /*
     * Move data memory location to another object
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
    void addToFileP(FILE *& ofile) const;
    void readFromFileP(FILE *& ifile);

    void writeToFile(const std::string& f)const;
    void readFromFile(const std::string& f);

    void cout()const;



    size_t sizeAt(size_t i)const;
    // higher dim row splits size_t sizeAt(size_t i,size_t j)const;
    // higher dim row splits size_t sizeAt(size_t i,size_t j, size_t k)const;
    // higher dim row splits size_t sizeAt(size_t i,size_t j, size_t k, size_t l)const;
    // higher dim row splits size_t sizeAt(size_t i,size_t j, size_t k, size_t l, size_t m)const;

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


    //static
    /**
     * assumes that the row splits are along the 1st dimension
     */
    static size_t findElementSplitLength(const std::vector<int64_t> & rowsplits,
            size_t nelements, size_t& startat, bool & exceeds_limit, bool sqelementslimit=false);

    /**
     * Split indices can directly be used with the split() function.
     * Returns e.g. {2,5,3,2}, which corresponds to DataSplitIndices of {2,7,10,12}
     */
    static std::vector<size_t>  getSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
            bool sqelementslimit=false, std::vector<bool>& size_ok=std::vector<bool>());

    /**
     * Split indices can directly be used with the split() function.
     * Returns e.g. {2,7,10,12} which corresponds to Split indices of {2,5,3,2}
     */
    static std::vector<size_t>  getDataSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
            bool sqelementslimit=false, std::vector<bool>& size_ok=std::vector<bool>());

    static std::vector<size_t>  dataSplitToSplitIndices(const std::vector<size_t>& data_splits);
    static std::vector<size_t>  splitToDataSplitIndices(const std::vector<size_t>& data_splits);

    static std::vector<int64_t> readRowSplitsFromFileP(FILE *& f, bool seeknext=true);


#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
    int isize() const {
        return (int)size_;
    }
    //does not transfer data ownership! only for quick writing etc.
    void assignFromNumpy(const boost::python::numpy::ndarray& ndarr,
            const boost::python::numpy::ndarray& rowsplits=boost::python::numpy::empty(
                    boost::python::make_tuple(0), boost::python::numpy::dtype::get_builtin<size_t>()));

    //copy data
    void createFromNumpy(const boost::python::numpy::ndarray& ndarr,
            const boost::python::numpy::ndarray& rowsplits=boost::python::numpy::empty(
                    boost::python::make_tuple(0), boost::python::numpy::dtype::get_builtin<size_t>()));

    //transfers data ownership and cleans simpleArray instance
    boost::python::tuple transferToNumpy(bool pad_rowsplits=false);

    //copy data
    boost::python::tuple copyToNumpy(bool pad_rowsplits=false)const;
#endif


private:
    size_t flatindex(size_t i, size_t j)const;
    size_t flatindex(size_t i, size_t j, size_t k)const;
    size_t flatindex(size_t i, size_t j, size_t k, size_t l)const;
    size_t flatindex(size_t i, size_t j, size_t k, size_t l, size_t m)const;
    size_t flatindex(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)const;

    std::vector<int64_t> padRowsplits()const;

    void copyFrom(const simpleArray<T>& a);
    void moveFrom(simpleArray<T> && a);
    size_t sizeFromShape(const std::vector<int>& shape) const;
    std::vector<int> shapeFromRowsplits()const; //split dim = 1!
    void checkShape(size_t ndims)const;
    void checkSize(size_t idx)const;
    void checkRaggedIndex(size_t i, size_t j)const;

    static std::vector<size_t>  priv_getSplitIndices(bool datasplit, const std::vector<int64_t> & rowsplits, size_t nelements_limit,
            bool sqelementslimit=false, std::vector<bool>& size_ok=std::vector<bool>());



#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
    std::vector<int> makeNumpyShape()const;
    void checkArray(const boost::python::numpy::ndarray& ndarr,
            boost::python::numpy::dtype dt=boost::python::numpy::dtype::get_builtin<T>())const;
    void fromNumpy(const boost::python::numpy::ndarray& ndarr,
                const boost::python::numpy::ndarray& rowsplits,
                bool copy);

#endif

    T * data_;
    std::vector<int> shape_;
    //this is int64 for better feeding to TF
    std::vector<int64_t> rowsplits_;
    size_t size_;
    bool assigned_;
};

template<class T>
simpleArray<T>::simpleArray() :
        data_(0), size_(0),assigned_(false) {
}

template<class T>
simpleArray<T>::simpleArray(std::vector<int> shape,const std::vector<int64_t>& rowsplits) :
        data_(0), size_(0),assigned_(false) {

    shape_ = shape;
    if(rowsplits.size()){
        if(rowsplits.size() != shape_.at(0)+1)
            throw std::runtime_error("simpleArray<T>::simpleArray: rowsplits.size() must equal shape[0] + 1");

        rowsplits_=rowsplits;
        shape_ = shapeFromRowsplits();
    }
    size_ = sizeFromShape(shape_);
    data_ = new T[size_];
}

template<class T>
simpleArray<T>::simpleArray(FILE *& ifile):simpleArray<T>(){
    readFromFileP(ifile);
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
    if (&a == this){
        return;}
    if (data_&& !assigned_)
        delete data_;
    data_ = a.data_;
    a.data_ = 0;
    assigned_ = a.assigned_;
    size_ = a.size_;
    a.size_ = 0;
    shape_ = std::move(a.shape_);
    a.shape_ = std::vector<int>();
    rowsplits_ = std::move(a.rowsplits_);
    a.rowsplits_= std::vector<int64_t>();
    a.clear();
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
    a.shape_ = std::vector<int>();
    rowsplits_ = std::move(a.rowsplits_);
    a.rowsplits_= std::vector<int64_t>();
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
void simpleArray<T>::setShape(std::vector<int> shape,const std::vector<int64_t>& rowsplits) {
    if(rowsplits.size()){
        *this = simpleArray<T>(shape,rowsplits);
    }
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
    return shape_.at(0);
}

template<class T>
T * simpleArray<T>::disownData() {
    T * dp = data_;
    data_ = 0;
    return dp;
}

/*
 * Splits on first axis.
 * Returns the first part, leaves the second
 * for ragged it is the number of elements index - need to be consistent with the rowplits
 *
 * add function 'size_t getClosestSplitPoint(size_t splitnelements, bool down=True)'
 *
 * for ragged, the split point is the INDEX IN THE ROWSPLIT VECTOR!
 *
 */
template<class T>
simpleArray<T> simpleArray<T>::split(size_t splitindex) {
    simpleArray<T> out;
    if (!shape_.size() || ( !isRagged() && splitindex > shape_.at(0))) {
        std::stringstream errMsg;
        errMsg << "simpleArray<T>::split: splitindex > shape_[0] : ";
        if(shape_.size())
            errMsg << splitindex << ", " << shape_.at(0);
        else
            errMsg <<"shape size: " << shape_.size() <<" empty array cannot be split.";
        cout();
        throw std::runtime_error(
                errMsg.str().c_str());
    }
    if(splitindex == shape_.at(0)){//exactly the whole array
        out = *this;
        clear();
        return out;
    }

    if(isRagged() && splitindex >  rowsplits_.size()){
        std::cout << "split index " << splitindex  << " range: " << rowsplits_.size()<< std::endl;
        throw std::runtime_error(
                "simpleArray<T>::split: ragged split index out of range");
    }


    //get split point for data
    ///insert rowsplit logic below
    int splitpoint = splitindex;
    if(isRagged()){
        splitpoint = rowsplits_.at(splitindex);
        for (size_t i = 2; i < shape_.size(); i++)
            splitpoint *= std::abs(shape_.at(i));
    }
    else{
        for (size_t i = 1; i < shape_.size(); i++)
            splitpoint *= std::abs(shape_.at(i));
    }


    size_t remaining = size_ - splitpoint;

    T * odata = new T[splitpoint];
    T * rdata = new T[remaining];

    memcpy(odata, data_, splitpoint * sizeof(T));
    memcpy(rdata, data_ + splitpoint, remaining * sizeof(T));
    if(!assigned_)
        delete data_;
    out.data_ = odata;
    data_ = rdata;
    ///insert rowsplit logic below
    out.shape_ = shape_;
    out.shape_.at(0) = splitindex;
    shape_.at(0) = shape_.at(0) - splitindex;
    if(isRagged()){

        out.rowsplits_ = std::vector<int64_t> (rowsplits_.begin(),rowsplits_.begin()+splitindex+1);
        int outnelements = out.rowsplits_.at(out.rowsplits_.size()-1);
        rowsplits_ = std::vector<int64_t> (rowsplits_.begin()+splitindex,rowsplits_.end());
        for(size_t i=0;i<rowsplits_.size();i++){
            rowsplits_.at(i)-=outnelements;
        }

        out.shape_ = out.shapeFromRowsplits();
        shape_ = shapeFromRowsplits();
    }
    ///
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
    if(isRagged() != a.isRagged())
        throw std::out_of_range(
                "simpleArray<T>::append: can't append ragged to non ragged or vice versa");

    std::vector<int> targetshape;
    if (shape_.size() > 1 && a.shape_.size() > 1) {
        size_t offset = 1;
        if(isRagged())
            offset = 2;

        std::vector<int> highshape = std::vector<int>(shape_.begin() + offset,
                shape_.end());
        std::vector<int> ahighshape = std::vector<int>(a.shape_.begin() + offset,
                a.shape_.end());
        if (highshape != ahighshape) {
            throw std::out_of_range(
                    "simpleArray<T>::append: all shapes but first axis must match");
        }
        targetshape.push_back(shape_.at(0) + a.shape_.at(0));
        if(isRagged())
            targetshape.push_back(-1);
        targetshape.insert(targetshape.end(), highshape.begin(),
                highshape.end());
    } else {
        targetshape.push_back(shape_.at(0) + a.shape_.at(0));
    }

    T * ndata = new T[size_ + a.size_];
    memcpy(ndata, data_, size_ * sizeof(T));
    memcpy(ndata + size_, a.data_, a.size_ * sizeof(T));
    if(!assigned_)
        delete data_;
    data_ = ndata;
    size_ = size_ + a.size_;
    ///insert rowsplit logic below
    shape_ = targetshape;
    //recalculate -XxY part of the shape
    //append the row splits if dimensions match (- on same axis)
    ///
    if(isRagged()){
        //need copy in case this == &a
        auto ars = a.rowsplits_;

        size_t oldrssize = rowsplits_.size();
        int nelements = rowsplits_.at(oldrssize-1);
        size_t newrssize = oldrssize + ars.size()-1;
        rowsplits_.resize(newrssize);
        for(size_t i=0;i<ars.size();i++){
            rowsplits_.at(i+oldrssize-1) = ars.at(i) + nelements;
        }

        shape_ = shapeFromRowsplits();//last
    }
}

template<class T>
void simpleArray<T>::addToFileP(FILE *& ofile) const {



    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);
    io::writeToFile(&size_, ofile);
    size_t ssize = shape_.size();
    io::writeToFile(&ssize, ofile);
    io::writeToFile(&shape_[0], ofile, shape_.size());

    size_t rssize = rowsplits_.size();
    io::writeToFile(&rssize,  ofile);

    if(rssize){
        quicklz<int64_t> iqlz;
        iqlz.writeCompressed(&rowsplits_[0],rssize , ofile);
    }
    quicklz<T> qlz;
    qlz.writeCompressed(data_, size_, ofile);

}

template<class T>
void simpleArray<T>::readFromFileP(FILE *& ifile) {
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
    rowsplits_ = std::vector<int64_t>(rssize, 0);

    if(rssize){
        quicklz<int64_t> iqlz;
        iqlz.readAll(ifile, &rowsplits_[0]);
    }
    quicklz<T> qlz;
    data_ = new T[size_];
    size_t nread = qlz.readAll(ifile, data_);
    if (nread != size_)
        throw std::runtime_error(
                "simpleArray<T>::readFromFile: expected and observed length don't match");

}



template<class T>
void simpleArray<T>::writeToFile(const std::string& f)const{
    FILE *ofile = fopen(f.data(), "wb");
    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);
    addToFileP(ofile);
    fclose(ofile);

}
template<class T>
void simpleArray<T>::readFromFile(const std::string& f){
    clear();
    FILE *ifile = fopen(f.data(), "rb");
    if(!ifile)
        throw std::runtime_error("simpleArray<T>::readFromFile: file "+f+" could not be opened.");
    float version = 0;
    io::readFromFile(&version, ifile);
    if(version != DJCDATAVERSION)
        throw std::runtime_error("simpleArray<T>::readFromFile: wrong format version");
    readFromFileP(ifile);
    fclose(ifile);
}

template<class T>
void simpleArray<T>::cout()const{
    std::cout << "data size "<< size() <<std::endl;
    for(int i=0;i<size();i++){
        std::cout << data()[i] << ", ";
    }
    std::cout << std::endl;
    for(const auto s: shape())
        std::cout << s << ", ";
    if(isRagged()){
        std::cout << "\nrow splits " << std::endl;
        if(rowsplits().size()){
            for(const auto s: rowsplits())
                std::cout << s << ", ";
        }
    }
    std::cout << std::endl;
}



template<class T>
 std::vector<size_t>  simpleArray<T>::priv_getSplitIndices(bool datasplit, const std::vector<int64_t> & rowsplits, size_t nelements_limit,
        bool sqelementslimit, std::vector<bool>& size_ok){

    std::vector<size_t> outIdxs;
    size_ok.clear();
    if(rowsplits.size()<1)
        return outIdxs;

    size_t i_old=0;
    size_t s_old = 0;
    size_t i_s = 0;
    while (true) {

        size_t s = rowsplits.at(i_s);
        size_t delta = s - s_old;
        size_t i_splitat = rowsplits.size()+1;

        if (sqelementslimit)
            delta *= delta;

        if (delta > nelements_limit && i_s != i_old+1) {
            i_splitat = i_s - 1;
            i_s--;
        }
        else if (delta == nelements_limit ||
                i_s == rowsplits.size() - 1 ||
                (delta > nelements_limit && i_s == i_old+1)) {
            i_splitat = i_s;
        }


        if (i_splitat < rowsplits.size()+1) {        //split
            bool is_good = (rowsplits.at(i_splitat) - rowsplits.at(i_old)
                    <= nelements_limit);
            size_ok.push_back(is_good);


            if(datasplit)
                outIdxs.push_back(i_splitat);
            else
                outIdxs.push_back(i_splitat - i_old);


            std::cout << "i_old " << i_old << "\n";
            std::cout << "i_s " << i_s << "\n";
            std::cout << "s_old " << s_old << "\n";
            std::cout << "s " << s << "\n";
            std::cout << "i_splitat " << i_splitat << "\n";
            std::cout << "is_good " << is_good << "\n";
            std::cout << "i_splitat - i_old " << i_splitat - i_old << "\n";
            std::cout << std::endl;

            i_old = i_splitat;
            s_old = rowsplits.at(i_old);
            //i_s = i_splitat;

        }
        i_s++;
        if(i_s == rowsplits.size())
            break;
    }

    return outIdxs;
}
/**
    * Split indices can directly be used with the split() function.
    * Returns e.g. {2,5,3,2}, which corresponds to DataSplitIndices of {2,7,10,12}
    */
template<class T>
std::vector<size_t>  simpleArray<T>::getSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
        bool sqelementslimit, std::vector<bool>& size_ok){
    return priv_getSplitIndices(false, rowsplits, nelements_limit, sqelementslimit,  size_ok);
}

/**
 * Split indices can directly be used with the split() function.
 * Returns e.g. {2,7,10,12} which corresponds to Split indices of {2,5,3,2}
 */

template<class T>
std::vector<size_t>  simpleArray<T>::getDataSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
        bool sqelementslimit, std::vector<bool>& size_ok){
    return priv_getSplitIndices(true, rowsplits, nelements_limit, sqelementslimit,  size_ok);
}

template<class T>
std::vector<size_t>  simpleArray<T>::dataSplitToSplitIndices(const std::vector<size_t>& data_splits){
    return std::vector<size_t>();
}

template<class T>
std::vector<size_t>  simpleArray<T>::splitToDataSplitIndices(const std::vector<size_t>& data_splits){
    return std::vector<size_t>();
}


template<class T>
std::vector<int64_t> simpleArray<T>::readRowSplitsFromFileP(FILE *& ifile, bool seeknext){

    float version = 0;
    size_t size;
    std::vector<int> shape;
    std::vector<int64_t> rowsplits;
    io::readFromFile(&version, ifile);
    if(version != DJCDATAVERSION)
        throw std::runtime_error("simpleArray<T>::readRowSplitsFromFileP: wrong format version");

    io::readFromFile(&size, ifile);

    size_t shapesize = 0;
    io::readFromFile(&shapesize, ifile);
    shape = std::vector<int>(shapesize, 0);
    io::readFromFile(&shape[0], ifile, shapesize);

    size_t rssize = 0;
    io::readFromFile(&rssize, ifile);
    rowsplits = std::vector<int64_t>(rssize, 0);

    if(rssize){
        quicklz<int64_t> iqlz;
        iqlz.readAll(ifile, &rowsplits[0]);
    }
    if(seeknext){
        quicklz<T> qlz;
        qlz.skipBlock(ifile);//sets file point to next item
    }
    return rowsplits;
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
size_t simpleArray<T>::sizeFromShape(const std::vector<int>& shape) const {
    int64_t size = 1;
    for (const auto s : shape){
        size *= std::abs(s);
        if(s<0)
            size=std::abs(s);//first ragged dimension has the full size of previous dimensions
    }
    return size;
}

template<class T>
std::vector<int> simpleArray<T>::shapeFromRowsplits()const{
    if(!isRagged()) return shape_;
    if(shape_.size()<2) return shape_;
    auto outshape = shape_;
    //rowsplits.size = nbatch+1
    outshape.at(1) = - (int)rowsplits_.at(rowsplits_.size()-1);
    return outshape;
}

template<class T>
void simpleArray<T>::checkShape(size_t ndims)const{
    //rowsplit ready due to definiton of shape
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
void simpleArray<T>::checkRaggedIndex(size_t i, size_t j)const{
    if(i > rowsplits_.size()-1 || j >= rowsplits_.at(i+1)-rowsplits_.at(i))
        throw std::out_of_range("simpleArray<T>::checkRaggedIndex: index out of range");
}



// rowsplit support being added here (see whiteboard)
template<class T>
size_t simpleArray<T>::flatindex(size_t i, size_t j)const{
    size_t flat = 0;
    if(isRagged()){
        checkRaggedIndex(i,j);
        flat = rowsplits_.at(i)+j;}
    else{
        flat = j + shape_.at(1)*i;}
    return flat;
}

//this can also be ragged
template<class T>
size_t simpleArray<T>::flatindex(size_t i, size_t j, size_t k)const{
    size_t flat = 0;
    if(isRagged()){
        checkRaggedIndex(i,j);
        flat = k + shape_.at(2)*(rowsplits_.at(i)+j);}
    else{
        flat = k + shape_.at(2)*(j + shape_.at(1)*i);}
    return flat;
}
template<class T>
size_t simpleArray<T>::flatindex(size_t i, size_t j, size_t k, size_t l)const{
    size_t flat = 0;
    if(isRagged()){
        checkRaggedIndex(i,j);
        flat = l + shape_.at(3)*(k + shape_.at(2)*(rowsplits_.at(i)+j));}
    else{
        flat = l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i));}
    return flat;
}
template<class T>
size_t simpleArray<T>::flatindex(size_t i, size_t j, size_t k, size_t l, size_t m)const{
    size_t flat = 0;
        if(isRagged()){
            checkRaggedIndex(i,j);
            flat = m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(rowsplits_.at(i)+j)));}
        else{
            flat = m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i)));}
    return flat;
}
template<class T>
size_t simpleArray<T>::flatindex(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)const{
    size_t flat = 0;
    if(isRagged()){
        checkRaggedIndex(i,j);
        flat = n + shape_.at(5)*(m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(rowsplits_.at(i)+j))));}
    else{
        flat = n + shape_.at(5)*(m + shape_.at(4)*(l + shape_.at(3)*(k + shape_.at(2)*(j + shape_.at(1)*i))));}
    return flat;
}


//no row split support here!! needs to be added!
//relatively easy if dimension 1 is row split. other dimensions harder


template<class T>
size_t simpleArray<T>::sizeAt(size_t i)const{
    checkShape(2);
    if(!isRagged())
        return shape_.at(1);
    checkRaggedIndex(i,0);
    return rowsplits_.at(i+1)-rowsplits_.at(i);
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
    size_t flat = flatindex(i,j);
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j)const{
    checkShape(2);
    size_t flat = flatindex(i,j);
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k){
    checkShape(3);
    size_t flat = flatindex(i,j,k);
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k)const{
    checkShape(3);
    size_t flat = flatindex(i,j,k);
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l){
    checkShape(4);
    size_t flat = flatindex(i,j,k,l);
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l)const{
    checkShape(4);
    size_t flat = flatindex(i,j,k,l);
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m){
    checkShape(5);
    size_t flat = flatindex(i,j,k,l,m);
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m)const{
    checkShape(5);
    size_t flat = flatindex(i,j,k,l,m);
    checkSize(flat);
    return data_[flat];
}

template<class T>
T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){
    checkShape(6);
    size_t flat = flatindex(i,j,k,l,m,n);
    checkSize(flat);
    return data_[flat];
}

template<class T>
const T & simpleArray<T>::at(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)const{
    checkShape(6);
    size_t flat = flatindex(i,j,k,l,m,n);
    checkSize(flat);
    return data_[flat];
}


#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
/*
 * PYTHON / NUMPY Interface below
 *
 */
template<class T>
std::vector<int> simpleArray<T>::makeNumpyShape()const{
    if(!isRagged())
        return shape_;
    std::vector<int> out;
    for(size_t i=1;i<shape_.size();i++)
        out.push_back(std::abs(shape_.at(i)));
    return out;
}

template<class T>
void simpleArray<T>::checkArray(const boost::python::numpy::ndarray& ndarr,
        boost::python::numpy::dtype dt)const{
    namespace p = boost::python;
    namespace np = boost::python::numpy;

    if(ndarr.get_dtype() != dt){
        std::string dts = p::extract<std::string>(p::str(ndarr.get_dtype()));
        std::string dtse = p::extract<std::string>(p::str(dt));
        std::cout <<"input has dtype "<< dts <<  " expected " << dtse<< std::endl;
        throw std::runtime_error("simpleArray<T>::checkArray: at least one array does not have right type. (e.g. row split must be int64)");
    }
    auto flags = ndarr.get_flags();
    if(!(flags & np::ndarray::CARRAY) || !(flags & np::ndarray::C_CONTIGUOUS)){
        throw std::runtime_error("simpleArray<T>::checkArray: at least one array is not C contiguous, please pass as numpy.ascontiguousarray(a, dtype='float32')");
    }
}

template<class T>
void simpleArray<T>::fromNumpy(const boost::python::numpy::ndarray& ndarr,
        const boost::python::numpy::ndarray& rowsplits, bool copy){
    namespace p = boost::python;
    namespace np = boost::python::numpy;

    clear();
    checkArray(ndarr, np::dtype::get_builtin<T>());

    T * npdata = (T*)(void*) ndarr.get_data();
    data_ = npdata;

    int ndim = ndarr.get_nd();
    std::vector<int> shape;
    for(int s=0;s<ndim;s++)
        shape.push_back(ndarr.shape(s));

    //check row splits, anyway copied
    if(len(rowsplits)>0){
        checkArray(rowsplits, np::dtype::get_builtin<int64_t>());
        rowsplits_.resize(len(rowsplits));
        memcpy(&(rowsplits_.at(0)),(int*)(void*) rowsplits.get_data(), rowsplits_.size() * sizeof(int64_t));
        shape.insert(shape.begin(),len(rowsplits)-1);
        shape_ = shape;
        shape_ = shapeFromRowsplits();
    }
    else{
        shape_ = shape;
    }
    size_ = sizeFromShape(shape_);

    if(copy){
        assigned_=false;
        data_ = new T[size_];
        memcpy(data_, npdata, size_* sizeof(T));
    }
    else{
        assigned_=true;
    }
}

template<class T>
void simpleArray<T>::assignFromNumpy(const boost::python::numpy::ndarray& ndarr,
        const boost::python::numpy::ndarray& rowsplits){
    fromNumpy(ndarr,rowsplits, false);
}
template<class T>
void simpleArray<T>::createFromNumpy(const boost::python::numpy::ndarray& ndarr,
        const boost::python::numpy::ndarray& rowsplits){
    fromNumpy(ndarr,rowsplits, true);
}


inline void destroyManagerCObject(PyObject* self) {
    auto * b = reinterpret_cast<float*>( PyCapsule_GetPointer(self, NULL) );
    delete [] b;
}

template<class T>
std::vector<int64_t> simpleArray<T>::padRowsplits()const{
    std::vector<int64_t>  out = rowsplits_;
    if(out.size()){
        size_t presize = rowsplits_.size();
        size_t nelements = rowsplits_.at(rowsplits_.size()-1);
        out.resize(nelements,0);
        out.at(out.size()-1) = presize;
    }
    return out;
}

//transfers data ownership and cleans simpleArray instance
template<class T>
boost::python::tuple simpleArray<T>::transferToNumpy(bool pad_rowsplits){
    namespace p = boost::python;
    namespace np = boost::python::numpy;

    auto shape = makeNumpyShape();
    T * data_ptr = disownData();

    np::ndarray dataarr = STLToNumpy<T>(data_ptr, shape, size(), false);
    if(pad_rowsplits){
        auto rsp = padRowsplits();
        np::ndarray rowsplits = STLToNumpy<int64_t>(&(rsp[0]), {(int)rsp.size()}, rsp.size(), true);
        clear();
        return p::make_tuple(dataarr,rowsplits);
    }
    //don't check. if rowsplits_.size()==0 function will return empty array and igonre invalid pointer
    np::ndarray rowsplits = STLToNumpy<int64_t>(&(rowsplits_[0]), {(int)rowsplits_.size()}, rowsplits_.size(), true);
    clear();//reset all
    return p::make_tuple(dataarr,rowsplits);
}

//cpoies data
template<class T>
boost::python::tuple simpleArray<T>::copyToNumpy(bool pad_rowsplits)const{

    namespace p = boost::python;
    namespace np = boost::python::numpy;

    auto shape = makeNumpyShape();
    T * data_ptr = data();

    np::ndarray dataarr = STLToNumpy<T>(data_ptr, shape, size(), true);
    if(pad_rowsplits){
        auto rsp = padRowsplits();
        np::ndarray rowsplits = STLToNumpy<int64_t>(&(rsp[0]), {(int)rsp.size()}, rsp.size(), true);
        return p::make_tuple(dataarr,rowsplits);
    }
    np::ndarray rowsplits = STLToNumpy<int64_t>(&(rowsplits_[0]), {(int)rowsplits_.size()}, rowsplits_.size(), true);
    return p::make_tuple(dataarr,rowsplits);

}


#endif

}

#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_ */
