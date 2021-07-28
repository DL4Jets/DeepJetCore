/*
 * simpleArray.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_

#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include "helper.h"
#include "pythonToSTL.h"

#include "c_helper.h"
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
#include <cmath>

namespace djc{


//has all non-data operations
class simpleArrayBase {
public:

    enum dtypes{float32,int32,undef};

    simpleArrayBase():size_(0),assigned_(false) {
    }
    virtual ~simpleArrayBase(){}

    simpleArrayBase(std::vector<int> shape,const std::vector<int64_t>& rowsplits = {});


  //  virtual simpleArrayBase& operator=(simpleArrayBase &&)=0;

    virtual void clear()=0;

    virtual void setShape(std::vector<int> shape,const std::vector<int64_t>& rowsplits = {})=0;

    virtual dtypes dtype()const{return undef;}
    int dtypeI()const{return (int)dtype();}
    std::string dtypeString()const{
        return dtypeToString(dtype());
    }

    void setName(const std::string& name){name_=name;}
    std::string name()const{return name_;}
    void setFeatureNames(const std::vector<std::string>& names){featnames_=names;}
    const std::vector<std::string>& featureNames()const{return featnames_;}

    virtual void fillZeros()=0;

    virtual void set(const size_t i, float val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, float val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, const size_t k, float val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, const size_t k, const size_t l, float val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, const size_t k, const size_t l, const size_t m, float val){throwWrongTypeSet();}

    virtual void set(const size_t i, int val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, int val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, const size_t k, int val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, const size_t k, const size_t l, int val){throwWrongTypeSet();}
    virtual void set(const size_t i, const size_t j, const size_t k, const size_t l, const size_t m, int val){throwWrongTypeSet();}

    static std::string dtypeToString(dtypes t);
    static dtypes stringToDtype(const std::string& s);

    const std::vector<int>& shape() const {
        return shape_;
    }

    virtual bool hasNanOrInf()const=0;

    boost::python::list shapePy()const;

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
    size_t getFirstDimension()const{
        if(!size_ || !shape_.size())
            return 0;
        return shape_.at(0);
    }

    const std::vector<int64_t>& rowsplits() const {
        return rowsplits_;
    }

    virtual void assignShape(std::vector<int> s)=0;


    virtual size_t validSlices(std::vector<size_t> splits)const=0;
    virtual bool validSlice(size_t splitindex_begin, size_t splitindex_end)const=0;

    virtual void addToFileP(FILE *& ofile) const=0;
    virtual void readFromFileP(FILE *& ifile,bool skip_data=false)=0;
    virtual void writeToFile(const std::string& f)const=0;
    virtual void readFromFile(const std::string& f)=0;

    void skipToNextArray(FILE *& ofile)const;
    /**
     * this goes back to the start of the header!
     */
    std::string readDtypeFromFileP(FILE *& ofile)const;
    std::string readDtypeFromFile(const std::string& f)const;

    dtypes readDtypeTypeFromFileP(FILE *& ofile)const;
    dtypes readDtypeTypeFromFile(const std::string& f)const;

    virtual void cout()const=0;

    virtual void append(const simpleArrayBase& )=0;

    /**
     * Split indices can directly be used with the split() function.
     * Returns elements e.g. {2,5,3,2}, which corresponds to DataSplitIndices of {2,7,10,12}
     */
    static std::vector<size_t>  getSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
            bool sqelementslimit, bool strict_limit, std::vector<bool>& size_ok, std::vector<size_t>& nelemtns_per_split);

    /**
     * Split indices can directly be used with the split() function.
     * Returns row splits e.g. {2,7,10,12} which corresponds to Split indices of {2,5,3,2}
     */
    static std::vector<size_t>  getDataSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
            bool sqelementslimit, bool strict_limit, std::vector<bool>& size_ok, std::vector<size_t>& nelemtns_per_split);

    /**
     * Transforms row splits to n_elements per ragged sample
     */
    static std::vector<int64_t>  dataSplitToSplitIndices(const std::vector<int64_t>& row_splits);

    /**
     * Transforms n_elements per ragged sample to row splits
     */
    static std::vector<int64_t>  splitToDataSplitIndices(const std::vector<int64_t>& data_splits);


    static std::vector<int64_t> readRowSplitsFromFileP(FILE *& f, bool seeknext=true);

    static std::vector<int64_t> mergeRowSplits(const std::vector<int64_t> & rowsplitsa, const std::vector<int64_t> & rowsplitsb);

    static std::vector<int64_t> splitRowSplits(std::vector<int64_t> & rowsplits, const size_t& splitpoint);


    int isize() const {
        return (int)size_;
    }




    //does not transfer data ownership! only for quick writing etc.
    virtual void assignFromNumpy(const boost::python::numpy::ndarray& ndarr,
            const boost::python::numpy::ndarray& rowsplits=boost::python::numpy::empty(
                    boost::python::make_tuple(0), boost::python::numpy::dtype::get_builtin<size_t>()))=0;

    //copy data
    virtual void createFromNumpy(const boost::python::numpy::ndarray& ndarr,
            const boost::python::numpy::ndarray& rowsplits=boost::python::numpy::empty(
                    boost::python::make_tuple(0), boost::python::numpy::dtype::get_builtin<size_t>()))=0;

    //transfers data ownership and cleans simpleArray instance
    virtual boost::python::tuple transferToNumpy(bool pad_rowsplits=false)=0;

    //copy data
    virtual boost::python::tuple copyToNumpy(bool pad_rowsplits=false)const=0;

    virtual void setFeatureNamesPy(boost::python::list l)=0;
    virtual boost::python::list featureNamesPy()=0;


protected:
    std::vector<int> shape_;
    std::string name_;
    std::vector<std::string> featnames_;
    //this is int64 for better feeding to TF
    std::vector<int64_t> rowsplits_;
    size_t size_;
    bool assigned_;


    size_t sizeFromShape(const std::vector<int>& shape) const;
    std::vector<int> shapeFromRowsplits()const; //split dim = 1!
    void checkShape(size_t ndims)const;
    void checkSize(size_t idx)const;
    void checkRaggedIndex(size_t i, size_t j)const;

    void getFlatSplitPoints(size_t splitindex_begin, size_t splitindex_end,
            size_t & splitpoint_start, size_t & splitpoint_end)const;

private:

    void throwWrongTypeSet()const{throw std::invalid_argument("simpleArrayBase::set: wrong data format");}

    static std::vector<size_t>  priv_getSplitIndices(bool datasplit, const std::vector<int64_t> & rowsplits, size_t nelements_limit,
                bool sqelementslimit, std::vector<bool>& size_ok, std::vector<size_t>& nelemtns_per_split, bool strict_limit);


};


template<class T>
class simpleArray: public simpleArrayBase { //inherits and implements data operations
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

    dtypes dtype()const;

    bool operator==(const simpleArray<T>& rhs)const;
    bool operator!=(const simpleArray<T>& rhs)const { return !(*this == rhs); }

    void clear();

    bool hasNanOrInf()const;

    //reshapes if possible, creates new else
    void setShape(std::vector<int> shape,const std::vector<int64_t>& rowsplits = {});

    T * data() const {
        return data_;
    }

    T * data() {
        return data_;
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

    simpleArray<T> getSlice(size_t splitindex_begin, size_t splitindex_end) const;

    /*
     *
     */
    size_t validSlices(std::vector<size_t> splits)const;
    bool validSlice(size_t splitindex_begin, size_t splitindex_end)const;


    simpleArray<T> shuffle(const std::vector<size_t>& shuffle_idxs)const;
    /*
     * appends along first axis
     * Cann append to an empty array (same as copy)
     */
    void append(const simpleArray<T>& a);
    void append(const simpleArrayBase& );



    /* file IO here
     * format: non compressed header (already writing rowsplits!):
     * size, shape.size(), [shape], rowsplits.size(), [rowsplits], compr: [data]
     *
     */
    void addToFileP(FILE *& ofile) const;
    void readFromFileP(FILE *& ifile,bool skip_data=false);

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

    void fillZeros();

    void set(const size_t i, T val){at(i)=val;}
    void set(const size_t i, const size_t j, T val){at(i,j)=val;}
    void set(const size_t i, const size_t j, const size_t k, T val){at(i,j,k)=val;}
    void set(const size_t i, const size_t j, const size_t k, const size_t l, T val){at(i,j,k,l)=val;}
    void set(const size_t i, const size_t j, const size_t k, const size_t l, const size_t m, T val){at(i,j,k,l,m)=val;}




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

    void setFeatureNamesPy(boost::python::list l);
    boost::python::list featureNamesPy();



private:
    size_t flatindex(size_t i, size_t j)const;
    size_t flatindex(size_t i, size_t j, size_t k)const;
    size_t flatindex(size_t i, size_t j, size_t k, size_t l)const;
    size_t flatindex(size_t i, size_t j, size_t k, size_t l, size_t m)const;
    size_t flatindex(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)const;

    std::vector<int64_t> padRowsplits()const;


    void copyFrom(const simpleArray<T>& a);
    void moveFrom(simpleArray<T> && a);




    std::vector<int> makeNumpyShape()const;
    void checkArray(const boost::python::numpy::ndarray& ndarr,
            boost::python::numpy::dtype dt=boost::python::numpy::dtype::get_builtin<T>())const;
    void fromNumpy(const boost::python::numpy::ndarray& ndarr,
                const boost::python::numpy::ndarray& rowsplits,
                bool copy);


    T * data_;
};

/* for later
template<class T>
class simpleArrayIndex {
public:
    simpleArrayIndex(simpleArray<T>& a, const int i):arr_(a){

    }
    simpleArrayIndex(const T&){
    //set value
    }

    operator T&() { return val; }
    operator T() const { return val; }

    simpleArrayIndex operator[](const int i){
        return simpleArrayIndex(arr_,i);
    }

private:
    simpleArray<T>& arr_;
    //some indexing
};
*/

template<class T>
simpleArray<T>::simpleArray() :
simpleArrayBase(),
        data_(0) {
}

template<class T>
simpleArray<T>::simpleArray(std::vector<int> shape,const std::vector<int64_t>& rowsplits) :
simpleArrayBase(shape,rowsplits) {
    data_ = new T[size_];
}

template<class T>
simpleArray<T>::simpleArray(FILE *& ifile):simpleArray<T>(){
    data_=0;
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
    data_=0;
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
    name_=a.name_;
    featnames_=a.featnames_;
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
    name_=a.name_;
    featnames_=a.featnames_;
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
bool simpleArray<T>::operator==(const simpleArray<T>& rhs)const{
    if(this == &rhs)
        return true;
    if(dtype() !=rhs.dtype())
        return false;
    if(name_!=rhs.name_)
        return false;
    if(featnames_!=rhs.featnames_)
        return false;
    if(size_!=rhs.size_)
        return false;
    if(shape_!=rhs.shape_)
        return false;
    if(rowsplits_!=rhs.rowsplits_)
        return false;
    //finally check data
    for(size_t i=0;i<size_;i++){
        if(data_[i]!=rhs.data_[i])
            return false;
    }
    return true;
}


template<class T>
void simpleArray<T>::clear() {
    if (data_&& !assigned_)
        delete data_;
    data_ = 0;
    shape_.clear();
    rowsplits_.clear();
    size_ = 0;
    name_="";
    featnames_.clear();
}

template<class T>
bool simpleArray<T>::hasNanOrInf()const{
    for(size_t i=0;i<size_;i++){
        if(std::isinf(data_[i]) || std::isnan(data_[i]))
            return true;
    }
    return false;
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
    size_t splitpoint = splitindex;
    if(isRagged()){
        splitpoint = rowsplits_.at(splitindex);
        for (size_t i = 2; i < shape_.size(); i++)
            splitpoint *= (size_t)std::abs(shape_.at(i));
    }
    else{
        for (size_t i = 1; i < shape_.size(); i++)
            splitpoint *= (size_t)std::abs(shape_.at(i));
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

        out.rowsplits_ = splitRowSplits(rowsplits_, splitindex);
        out.shape_ = out.shapeFromRowsplits();
        shape_ = shapeFromRowsplits();
    }
    ///
    out.size_ = sizeFromShape(out.shape_);
    size_ = sizeFromShape(shape_);
    out.featnames_ = featnames_;
    out.name_ = name_;
    return out;
}



template<class T>
simpleArray<T> simpleArray<T>::getSlice(size_t splitindex_begin, size_t splitindex_end) const{
    simpleArray<T> out;
    if (!shape_.size() || ( !isRagged() && (splitindex_end > shape_.at(0) || splitindex_begin > shape_.at(0))) ) {
        std::stringstream errMsg;
        errMsg << "simpleArray<T>::slice: splitindex_end > shape_[0] : ";
        if(shape_.size())
            errMsg << splitindex_end << ", " << shape_.at(0);
        else
            errMsg <<"shape size: " << shape_.size() <<" empty array cannot be split.";
        cout();
        throw std::runtime_error(
                errMsg.str().c_str());
    }
    if(splitindex_end == shape_.at(0) && splitindex_begin==0){//exactly the whole array
        out = *this;
        return out;
    }

    if(isRagged() && (splitindex_end >=  rowsplits_.size() || splitindex_begin>= rowsplits_.size())){
        std::cout << "split index " << splitindex_end  << " - "<< splitindex_begin<< " allowed: " << rowsplits_.size()<< std::endl;
        throw std::runtime_error(
                "simpleArray<T>::slice: ragged split index out of range");
    }
    if(splitindex_end == splitindex_begin){
        //throw std::runtime_error("simpleArray<T>::slice: attempt to create empty slice");
        //actually, allow this here and let the problem be handled further down the line, just throw warning for now
        std::cout << "simpleArray<T>::slice: WARNING: attempt to create empty slice at "<< splitindex_begin <<std::endl;
    }

    //for arrays larger than 8/16(?) GB, size_t is crucial
    size_t splitpoint_start, splitpoint_end;
    getFlatSplitPoints(splitindex_begin,splitindex_end,
            splitpoint_start, splitpoint_end );

    T * odata = new T[splitpoint_end-splitpoint_start];
    memcpy(odata, data_+splitpoint_start, (splitpoint_end-splitpoint_start) * sizeof(T));

    out.data_ = odata;

    out.shape_ = shape_;
    out.shape_.at(0) = splitindex_end-splitindex_begin;

    if(isRagged()){
        auto rscopy = rowsplits_;
        rscopy = splitRowSplits(rscopy, splitindex_end);
        splitRowSplits(rscopy, splitindex_begin);
        out.rowsplits_ = rscopy;
        out.shape_ = out.shapeFromRowsplits();
    }
    ///
    out.size_ = sizeFromShape(out.shape_);
    out.name_ = name_;
    out.featnames_ = featnames_;

    return out;

}


template<class T>
size_t simpleArray<T>::validSlices(std::vector<size_t> splits)const{
    size_t out=0;
    if(!isRagged()){
        while(splits.at(out) <= shape_.at(0) && out< splits.size())
            out++;
        return out;
    }
    else{
        while(splits.at(out) < rowsplits_.size() && out < splits.size())
            out++;
        return out;
    }
}

template<class T>
bool simpleArray<T>::validSlice(size_t splitindex_begin, size_t splitindex_end)const{
    if (!shape_.size() || ( !isRagged() && (splitindex_end > shape_.at(0) || splitindex_begin > shape_.at(0))) )
        return false;
    if(isRagged() && (splitindex_end >=  rowsplits_.size() || splitindex_begin>= rowsplits_.size()))
        return false;
    return true;
}



template<class T>
simpleArray<T> simpleArray<T>::shuffle(const std::vector<size_t>& shuffle_idxs)const{
    //check
    bool isvalid = true;
    for(const auto& idx: shuffle_idxs){
        isvalid &= validSlice(idx,idx+1);
    }
    if(!isvalid)
        throw std::runtime_error("simpleArray<T>::shuffle: indices not valid");

    //copy data
    auto out=*this;
    size_t next=0;
    for(const auto idx: shuffle_idxs){

        size_t source_splitpoint_start, source_splitpoint_end;
        getFlatSplitPoints(idx,idx+1,
                source_splitpoint_start, source_splitpoint_end );
        size_t n_elements = source_splitpoint_end-source_splitpoint_start;
        memcpy(out.data_+next,
                data_+source_splitpoint_start,n_elements  * sizeof(T));

        next+=n_elements;
    }
    //recreate row splits
    if(isRagged()){
        auto nelems = dataSplitToSplitIndices(rowsplits_);
        auto new_nelems=nelems;
        for(size_t i=0;i<shuffle_idxs.size();i++)
            new_nelems.at(i)=nelems.at(shuffle_idxs.at(i));
        out.rowsplits_ = splitToDataSplitIndices(new_nelems);
    }
    return out;
}


/*
 * Merges along first axis
 */
template<class T>
void simpleArray<T>::append(const simpleArray<T>& a) {

    if (!data_ && size_ == 0) {
        //just save feature names and name
        auto namesv = name_;
        auto fnamesv = featnames_;
        *this = a;
        name_=namesv;
        featnames_ = fnamesv;
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

        rowsplits_ = mergeRowSplits(rowsplits_, ars);

        shape_ = shapeFromRowsplits();//last
    }
}

template<class T>
void simpleArray<T>::append(const simpleArrayBase& arr){
    if(dtype() != arr.dtype())
        throw std::runtime_error("simpleArray<T>::append: needs to be same dtype");
    append(dynamic_cast<const simpleArray<T> &>(arr));
}


template<class T>
void simpleArray<T>::addToFileP(FILE *& ofile) const {

    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);
    auto tdtype = dtype();
    io::writeToFile(&tdtype, ofile);
    io::writeToFile(&name_, ofile);
    io::writeToFile(&featnames_, ofile);
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
void simpleArray<T>::readFromFileP(FILE *& ifile, bool skip_data) {
    clear();

    float version = 0;
    io::readFromFile(&version, ifile);

    if(!checkVersionCompatible(version)){
        throw std::runtime_error("simpleArray<T>::readFromFile: wrong format version");
    }
    dtypes rdtype=dtype();
    if(checkVersionStrict(version)){
        io::readFromFile(&rdtype, ifile);
        io::readFromFile(&name_, ifile);
        io::readFromFile(&featnames_, ifile);
    }
    if(rdtype!=dtype())
        throw std::runtime_error("simpleArray<T>::readFromFileP: wrong dtype");

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
    if(skip_data){
        if(rowsplits_.size())
            rowsplits_={(int64_t)0};
        else
            rowsplits_.clear();
        data_=0;
        size_=0;
        shape_.at(0)=0;
        qlz.skipBlock(ifile);
        return;
    }

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
    if(!checkVersionCompatible(version))
        throw std::runtime_error("simpleArray<T>::readFromFile: wrong format version: "+std::to_string(version));
    readFromFileP(ifile);
    fclose(ifile);
}

template<class T>
void simpleArray<T>::cout()const{
    std::cout << "name: " << name_ << std::endl;
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
    std::cout << "data size "<< size() <<std::endl;
    std::cout << "feature names "<< featureNames() <<std::endl;
    std::cout << std::endl;
}




template<class T>
void simpleArray<T>::copyFrom(const simpleArray<T>& a) {

    if (&a == this) {
        return;
    }
    if (data_&& !assigned_)
        delete data_;
    name_=a.name_;
    featnames_=a.featnames_;
    data_ = new T[a.size_];
    memcpy(data_, a.data_, a.size_ * sizeof(T));

    size_ = a.size_;
    shape_ = a.shape_;
    rowsplits_ = a.rowsplits_;
    assigned_=false;
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


template<class T>
void simpleArray<T>::fillZeros(){
    for(size_t i=0;i<size();i++)
        data_[i]=0;
}


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
        memcpy(&(rowsplits_.at(0)),(int64_t*)(void*) rowsplits.get_data(), rowsplits_.size() * sizeof(int64_t));
        //check if row splits make sense
        if(shape.at(0) != rowsplits_.at(rowsplits_.size()-1)){
            throw std::out_of_range("simpleArray<T>::fromNumpy: row splits and input array incompatible. rowsplits[-1] != arr.shape[0].");
        }
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
std::vector<int64_t> simpleArray<T>::padRowsplits()const{ //rs 0, 1, 1 element
    std::vector<int64_t>  out = rowsplits_;
    if(out.size()){
        size_t presize = rowsplits_.size();
        size_t nelements = rowsplits_.at(rowsplits_.size()-1);
        if((nelements<1 && !shape_.size()) || nelements!=-shape_.at(1)){
            throw std::runtime_error("simpleArray<T>::padRowsplits: rowsplit format seems broken. Total number of entries at last entry: "+
                    to_str(nelements)+" row splits: "+to_str(rowsplits_)+ " versus actual shape "+to_str(shape_));
        }
        if(nelements<3)//keep format of [rs ], nelements
            nelements=3;
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

template<class T>
void simpleArray<T>::setFeatureNamesPy(boost::python::list l){
    std::vector<std::string> names = toSTLVector<std::string>(l);
    setFeatureNames(names);
}
template<class T>
boost::python::list simpleArray<T>::featureNamesPy(){
    boost::python::list l;
    for(const auto& v:featureNames())
        l.append(v);
    return l;
}



typedef simpleArray<float> simpleArray_float32;
typedef simpleArray<int32_t> simpleArray_int32;



}//namespace

#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAY_H_ */
