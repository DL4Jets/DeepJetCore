/*
 * trainDataInterface.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_
#define DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_

#include "simpleArray.h"
#include <stdio.h>
#include "IO.h"

namespace djc{

/*
 * The idea is to make this a fixed size array class, that is filled with data and then written out once full.
 * a truncate function will allow to  truncate arrays at a given position.
 * This is memory intense, but can be written out in small pieces and then merged
 *
 * No checks on the first dimension because of possibly ragged arrays
 */
template<class T>
class trainData{
public:

    //just give access to the vectors, don't wrap like crazy

    const size_t addFeatureArray(std::vector<int> shape);
    const size_t addTruthArray(std::vector<int> shape);
    const size_t addWeightArray(std::vector<int> shape);

    const simpleArray<T> & featureArray(size_t idx) const {
        return feature_arrays_.at(idx);
    }

    const simpleArray<T> & truthArray(size_t idx) const {
        return truth_arrays_.at(idx);
    }

    const simpleArray<T> & weightArray(size_t idx) const {
        return weight_arrays_.at(idx);
    }

    simpleArray<T> & featureArray(size_t idx)  {
        return feature_arrays_.at(idx);
    }

    simpleArray<T> & truthArray(size_t idx)  {
        return truth_arrays_.at(idx);
    }

    simpleArray<T> & weightArray(size_t idx)  {
        return weight_arrays_.at(idx);
    }

    size_t nFeatureArrays()const{return feature_arrays_.size();}
    size_t nTruthArrays()const{return truth_arrays_.size();}
    size_t nWeightArrays()const{return weight_arrays_.size();}

    /*
     * truncate all along first axis
     */
    void truncate(size_t position);

    /*
     * append along first axis
     */
    void append(const trainData<T>& );

    /*
     * split along first axis
     * Returns the second part, leaves the first.
     */
    trainData<T> split(size_t splitindex);

    size_t nElements()const{
        if(feature_arrays_.size())
            return feature_arrays_.at(0).getFirstDimension();
        else
            return 0;
    }

    void writeToFile(std::string filename)const;

    void readFromFile(std::string filename);

    void clear();

private:
    void writeArrayVector(const std::vector<simpleArray<T> >&, FILE *&) const;
    std::vector<simpleArray<T> > readArrayVector(FILE *&) const;
    std::vector<simpleArray<T> > feature_arrays_;
    std::vector<simpleArray<T> > truth_arrays_;
    std::vector<simpleArray<T> > weight_arrays_;

};


template<class T>
const size_t trainData<T>::addFeatureArray(std::vector<int> shape){
    size_t idx = feature_arrays_.size();
    feature_arrays_.push_back(simpleArray<T>(shape));
    return idx;
}


template<class T>
const size_t trainData<T>::addTruthArray(std::vector<int> shape){
    size_t idx = truth_arrays_.size();
    truth_arrays_.push_back(simpleArray<T>(shape));
    return idx;
}

template<class T>
const size_t trainData<T>::addWeightArray(std::vector<int> shape){
    size_t idx = weight_arrays_.size();
    weight_arrays_.push_back(simpleArray<T>(shape));
    return idx;
}

/*
 * truncate all along first axis
 */
template<class T>
void trainData<T>::truncate(size_t position){
    *this = split(position);
}

/*
 * append along first axis
 */
template<class T>
void trainData<T>::append(const trainData<T>& td) {
    //allow empty append
    if (!feature_arrays_.size() && !truth_arrays_.size()
            && !weight_arrays_.size()) {
        *this = td;
        return;
    }
    if (feature_arrays_.size() != td.feature_arrays_.size()
            || truth_arrays_.size() != td.truth_arrays_.size()
            || weight_arrays_.size() != td.weight_arrays_.size()) {
        throw std::out_of_range("trainData<T>::append: format not compatible.");
    }
    for(size_t i=0;i<feature_arrays_.size();i++)
        feature_arrays_.at(i).append(td.feature_arrays_.at(i));
    for(size_t i=0;i<truth_arrays_.size();i++)
        truth_arrays_.at(i).append(td.truth_arrays_.at(i));
    for(size_t i=0;i<weight_arrays_.size();i++)
        weight_arrays_.at(i).append(td.weight_arrays_.at(i));
}

/*
 * split along first axis
 * Returns the first part, leaves the second.
 */
template<class T>
trainData<T> trainData<T>::split(size_t splitindex) {
    trainData<T> out;
    for (auto& a : feature_arrays_)
        out.feature_arrays_.push_back(a.split(splitindex));
    for (auto& a : truth_arrays_)
        out.truth_arrays_.push_back(a.split(splitindex));
    for (auto& a : weight_arrays_)
        out.weight_arrays_.push_back(a.split(splitindex));

    return out;
}

template<class T>
void trainData<T>::writeToFile(std::string filename)const{

    FILE *ofile = fopen(filename.data(), "wb");
    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);
    writeArrayVector(feature_arrays_,ofile);
    writeArrayVector(truth_arrays_,ofile);
    writeArrayVector(weight_arrays_,ofile);
    fclose(ofile);
}

template<class T>
void trainData<T>::readFromFile(std::string filename){
    clear();
    FILE *ifile = fopen(filename.data(), "rb");
    if(!ifile)
        throw std::runtime_error("trainData<T>::readFromFile: file "+filename+" could not be opened.");
    float version = 0;
    io::readFromFile(&version, ifile);
    if(version != DJCDATAVERSION)
        throw std::runtime_error("trainData<T>::readFromFile: wrong format version");

    feature_arrays_ = readArrayVector(ifile);
    truth_arrays_ = readArrayVector(ifile);
    weight_arrays_ = readArrayVector(ifile);
    fclose(ifile);

}

template<class T>
void trainData<T>::clear() {
    feature_arrays_.clear();
    truth_arrays_.clear();
    weight_arrays_.clear();
}

template<class T>
void trainData<T>::writeArrayVector(const std::vector<simpleArray<T> >& v, FILE *& ofile) const{

    size_t size = v.size();
    io::writeToFile(&size, ofile);
    for(const auto& a: v)
        a.addToFile(ofile);

}
template<class T>
std::vector<simpleArray<T> > trainData<T>::readArrayVector(FILE *& ifile) const{
    std::vector<simpleArray<T> >  out;
    size_t size = 0;
    io::readFromFile(&size, ifile);
    for(size_t i=0;i<size;i++)
        out.push_back(simpleArray<T> (ifile));
    return out;
}

/*
 * Array storage:
 * length, shape, length row splits, [row splits] ? numpy doesn't like ragged... maybe just return row splits?
 * (shape is int. negative entries provoke row splits, only splits in one dimension supported)
 *
 * all data is float32. only row splits and shapes should be int (not size_t) for simple python conversion
 *
 * make it a traindata object
 *
 * interface:
 *
 * writeToFile(vector< float * > c_arrays (also pointers to first vec element), vector< vector<int> > shapes, (opt)  vector< vector<int> > row_splits, filename)
 *
 * readFromFile
 *
 */

/*
 *
 * Make a write CPP interface that does not need boost whatsoever!
 * Then wrap it for python-numpy bindings externally
 *
 *
 */

/*
 * uncompressed header with all shape infos
 * compressed x,y,w lists or arrays?
 *
 *
 */

}//namespace

#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_ */
