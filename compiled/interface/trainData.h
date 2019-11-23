/*
 * trainDataInterface.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_
#define DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_

//#define DJC_DATASTRUCTURE_PYTHON_BINDINGS//DEBUG

#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include "helper.h"
#endif

#include "simpleArray.h"
#include <stdio.h>
#include "IO.h"

#include <iostream>

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


    //takes ownership
    int storeFeatureArray( simpleArray<T>&);
    int storeTruthArray( simpleArray<T>&);
    int storeWeightArray( simpleArray<T>&);

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

    int nFeatureArrays()const{return feature_arrays_.size();}
    int nTruthArrays()const{return truth_arrays_.size();}
    int nWeightArrays()const{return weight_arrays_.size();}

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
        if(feature_shapes_.size() && feature_shapes_.at(0).size())
            return feature_shapes_.at(0).at(0);
        else
            return 0;
    }

    const std::vector<std::vector<int> > & featureShapes()const{return  feature_shapes_;}
    const std::vector<std::vector<int> > & truthShapes()const{return  truth_shapes_;}
    const std::vector<std::vector<int> > & weightShapes()const{return  weight_shapes_;}

    void writeToFile(std::string filename)const;

    void readFromFile(std::string filename);

    //could use a readshape or something!
    void readShapesFromFile(const std::string& filename);

    std::vector<int64_t> readShapesAndRowSplitsFromFile(const std::string& filename, bool checkConsistency=true);

    void clear();


    //from python
    void skim(size_t batchelement);


#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS

    boost::python::list getKerasFeatureShapes()const;
    // not needed boost::python::list getKerasTruthShapes()const;
    // not needed boost::python::list getKerasWeightShapes()const;

    //data generator interface get back numpy arrays / tf.tensors here for keras feeding!

    boost::python::list getTruthRaggedFlags()const;

    //no ragged support
    boost::python::list transferFeatureListToNumpy();

    //has ragged support
    boost::python::list transferTruthListToNumpy();

    //no ragged support
    boost::python::list transferWeightListToNumpy();

#endif

private:
    void checkFile(FILE *& f, const std::string& filename="")const;

    void writeArrayVector(const std::vector<simpleArray<T> >&, FILE *&) const;
    std::vector<simpleArray<T> > readArrayVector(FILE *&) const;
    void readRowSplitArray(FILE *&, std::vector<int64_t> &rs, bool check)const;
    std::vector<std::vector<int> > getShapes(const std::vector<simpleArray<T> >& a)const;
    template <class U>
    void writeNested(const std::vector<std::vector<U> >& v, FILE *&)const;
    template <class U>
    void readNested( std::vector<std::vector<U> >& v, FILE *&)const;

    void updateShapes();

    std::vector<simpleArray<T> > feature_arrays_;
    std::vector<simpleArray<T> > truth_arrays_;
    std::vector<simpleArray<T> > weight_arrays_;

    std::vector<std::vector<int> > feature_shapes_;
    std::vector<std::vector<int> > truth_shapes_;
    std::vector<std::vector<int> > weight_shapes_;

};


template<class T>
int trainData<T>::storeFeatureArray(simpleArray<T> & a){
    size_t idx = feature_arrays_.size();
    feature_arrays_.push_back(std::move(a));
    a.clear();
    updateShapes();
    return idx;
}


template<class T>
int trainData<T>::storeTruthArray(simpleArray<T>& a){
    size_t idx = truth_arrays_.size();
    truth_arrays_.push_back(std::move(a));
    a.clear();
    updateShapes();
    return idx;
}

template<class T>
int trainData<T>::storeWeightArray(simpleArray<T> & a){
    size_t idx = weight_arrays_.size();
    weight_arrays_.push_back(std::move(a));
    a.clear();
    updateShapes();
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
    if(!td.feature_arrays_.size() && !td.truth_arrays_.size()
            && !td.weight_arrays_.size()){
        return ; //nothing to do
    }
    if (feature_arrays_.size() != td.feature_arrays_.size()
            || truth_arrays_.size() != td.truth_arrays_.size()
            || weight_arrays_.size() != td.weight_arrays_.size()) {
        std::cout << "nfeat " << feature_arrays_.size() << "-" << td.feature_arrays_.size() <<'\n'
                << "ntruth " << truth_arrays_.size() << "-" << td.truth_arrays_.size()<<'\n'
                << "nweights " << weight_arrays_.size() << "-" <<  td.weight_arrays_.size() <<std::endl;
        throw std::out_of_range("trainData<T>::append: format not compatible.");
    }
    for(size_t i=0;i<feature_arrays_.size();i++)
        feature_arrays_.at(i).append(td.feature_arrays_.at(i));
    for(size_t i=0;i<truth_arrays_.size();i++)
        truth_arrays_.at(i).append(td.truth_arrays_.at(i));
    for(size_t i=0;i<weight_arrays_.size();i++)
        weight_arrays_.at(i).append(td.weight_arrays_.at(i));
    updateShapes();
}

/*
 * split along first axis
 * Returns the first part, leaves the second.
 *
 * Can use some performance improvements
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

    updateShapes();
    out.updateShapes();
    return out;
}

template<class T>
void trainData<T>::writeToFile(std::string filename)const{

    FILE *ofile = fopen(filename.data(), "wb");
    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);

    //shape infos only
    writeNested(getShapes(feature_arrays_), ofile);
    writeNested(getShapes(truth_arrays_), ofile);
    writeNested(getShapes(weight_arrays_), ofile);

    //data
    writeArrayVector(feature_arrays_, ofile);
    writeArrayVector(truth_arrays_, ofile);
    writeArrayVector(weight_arrays_, ofile);
    fclose(ofile);

}

template<class T>
void trainData<T>::readFromFile(std::string filename){
    clear();
    FILE *ifile = fopen(filename.data(), "rb");
    checkFile(ifile);
    readNested(feature_shapes_, ifile);
    readNested(truth_shapes_, ifile);
    readNested(weight_shapes_, ifile);

    feature_arrays_ = readArrayVector(ifile);
    truth_arrays_ = readArrayVector(ifile);
    weight_arrays_ = readArrayVector(ifile);

    fclose(ifile);

}

template<class T>
void trainData<T>::readShapesFromFile(const std::string& filename){

    FILE *ifile = fopen(filename.data(), "rb");
    checkFile(ifile,filename);

    readNested(feature_shapes_, ifile);
    readNested(truth_shapes_, ifile);
    readNested(weight_shapes_, ifile);

    fclose(ifile);

}

template<class T>
std::vector<int64_t> trainData<T>::readShapesAndRowSplitsFromFile(const std::string& filename, bool checkConsistency){
    std::vector<int64_t> rowsplits;

    FILE *ifile = fopen(filename.data(), "rb");
    checkFile(ifile,filename);

    //shapes
    std::vector<std::vector<int> > dummy;
    readNested(feature_shapes_, ifile);
    readNested(truth_shapes_, ifile);
    readNested(weight_shapes_, ifile);

    //features
    readRowSplitArray(ifile,rowsplits,checkConsistency);
    if(!checkConsistency && rowsplits.size()){
        fclose(ifile);
        return rowsplits;
    }
    //truth
    readRowSplitArray(ifile,rowsplits,checkConsistency);
    if(!checkConsistency && rowsplits.size()){
        fclose(ifile);
        return rowsplits;
    }
    //weights
    readRowSplitArray(ifile,rowsplits,checkConsistency);

    fclose(ifile);
    return rowsplits;

}

template<class T>
void trainData<T>::clear() {
    feature_arrays_.clear();
    truth_arrays_.clear();
    weight_arrays_.clear();
}

template<class T>
void trainData<T>::checkFile(FILE *& ifile, const std::string& filename)const{
    if(!ifile)
        throw std::runtime_error("trainData<T>::readFromFile: file "+filename+" could not be opened.");
    float version = 0;
    io::readFromFile(&version, ifile);
    if(version != DJCDATAVERSION)
        throw std::runtime_error("trainData<T>::readFromFile: wrong format version");

}

template<class T>
void trainData<T>::writeArrayVector(const std::vector<simpleArray<T> >& v, FILE *& ofile) const{

    size_t size = v.size();
    io::writeToFile(&size, ofile);
    for(const auto& a: v)
        a.addToFileP(ofile);

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

template<class T>
void trainData<T>::readRowSplitArray(FILE *& ifile, std::vector<int64_t> &rowsplits, bool check)const{
    size_t size = 0;
    io::readFromFile(&size, ifile);
    for(size_t i=0;i<size;i++){
        auto frs = simpleArray<T>::readRowSplitsFromFileP(ifile, true);
        if(frs.size()){
            if(check){
                if(rowsplits.size() && rowsplits != frs)
                    throw std::runtime_error("trainData<T>::readShapesAndRowSplitsFromFile: row splits inconsistent");
            }
            rowsplits=frs;
        }
    }
}

template<class T>
std::vector<std::vector<int> > trainData<T>::getShapes(const std::vector<simpleArray<T> >& a)const{
    std::vector<std::vector<int> > out;
    for(const auto& arr: a)
        out.push_back(arr.shape());
    return out;
}

template<class T>
template <class U>
void trainData<T>::writeNested(const std::vector<std::vector<U> >& v, FILE *& ofile)const{

    size_t size = v.size();
    io::writeToFile(&size, ofile);
    for(size_t i=0;i<size;i++){
        size_t nsize = v.at(i).size();
        io::writeToFile(&nsize, ofile);
        if(nsize==0)
            continue;
        io::writeToFile(&(v.at(i).at(0)),ofile,nsize);
    }

}

template<class T>
template <class U>
void trainData<T>::readNested(std::vector<std::vector<U> >& v, FILE *& ifile)const{

    size_t size = 0;
    io::readFromFile(&size, ifile);
    v.resize(size,std::vector<U>(0));
    for(size_t i=0;i<size;i++){
        size_t nsize = 0;
        io::readFromFile(&nsize, ifile);
        v.at(i).resize(nsize);
        if(nsize==0)
            continue;
        io::readFromFile(&(v.at(i).at(0)),ifile,nsize);
    }

}

template<class T>
void trainData<T>::updateShapes(){

    feature_shapes_ = getShapes(feature_arrays_);
    truth_shapes_ = getShapes(truth_arrays_);
    weight_shapes_ = getShapes(weight_arrays_);

}

template<class T>
void trainData<T>::skim(size_t batchelement){
    if(batchelement > nElements())
        throw std::out_of_range("trainData<T>::skim: batch element out of range");
    for(auto & a : feature_arrays_){
        a.split(batchelement);
        a.split(1);
    }
    for(auto & a : truth_arrays_){
        a.split(batchelement);
        a.split(1);
    }
    for(auto & a : weight_arrays_){
        a.split(batchelement);
        a.split(1);
    }
    updateShapes();
}


#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS

template<class T>
boost::python::list trainData<T>::getKerasFeatureShapes()const{
    boost::python::list out;
    for(const auto& a: feature_shapes_){
        boost::python::list nlist;
        bool wasragged=false;
        for(size_t i=1;i<a.size();i++){
            if(a.at(i)<0){
                nlist = boost::python::list();//ignore everything before
                wasragged=true;
            }
            else
                nlist.append(std::abs(a.at(i)));
        }
        out.append(nlist);
        if(wasragged){
            boost::python::list rslist;
            rslist.append(1);
            out.append(rslist);
        }
    }
    return out;
}
//template<class T>
//boost::python::list trainData<T>::getKerasTruthShapes()const{
//    boost::python::list out;
//    for(const auto& a: truth_arrays_){
//        size_t start=1;
//        if(a.isRagged())
//            start=2;
//        for(size_t i=start;i<a.shape().size();i++)
//            out.append(std::abs(a.shape().at(i)));
//    }
//    return out;
//}
//template<class T>
//boost::python::list trainData<T>::getKerasWeightShapes()const{
//    boost::python::list out;
//    for(const auto& a: weight_shapes_){
//        for(size_t i=1;i<a.size();i++){
//            if(a.at(i)<0){
//                out = boost::python::list();//igonre everything before
//            }
//            out.append(std::abs(a.at(i)));
//        }
//    }
//    return out;
//}


template<class T>
boost::python::list trainData<T>::getTruthRaggedFlags()const{
    boost::python::list out;
    for(const auto& a: truth_shapes_){
        bool isragged = false;
        for(const auto & s: a)
            if(s<0){
                isragged=true;
                break;
            }
        if(isragged)
            out.append(true);
        else
            out.append(false);
    }
    return out;
}

template<class T>
boost::python::list trainData<T>::transferFeatureListToNumpy(){
    namespace p = boost::python;
    namespace np = boost::python::numpy;
    p::list out;
    for( auto& a: feature_arrays_){
        if(a.isRagged()){
            auto arrt = a.transferToNumpy(true);//pad row splits
            out.append(arrt[0]);//data
            np::ndarray rs = boost::python::extract<np::ndarray>(arrt[1]);
            out.append(rs.reshape(p::make_tuple(-1,1)));//row splits
        }
        else
            out.append(a.transferToNumpy(true)[0]);
    }
    return out;
}

template<class T>
boost::python::list trainData<T>::transferTruthListToNumpy(){
    namespace p = boost::python;
    namespace np = boost::python::numpy;

    boost::python::list out;
        for( auto& a: truth_arrays_){
            if(a.isRagged()){
                //auto arrt = a.transferToNumpy(false);
                //boost::python::list subl;
                //subl.append(arrt[0]);
                //subl.append(arrt[1]);
                //out.append(subl);
                auto arrt = a.transferToNumpy(true);//pad row splits
                out.append(arrt[0]);//data
                np::ndarray rs = boost::python::extract<np::ndarray>(arrt[1]);
                out.append(rs.reshape(p::make_tuple(-1,1)));//row splits
            }
            else{
                out.append(a.transferToNumpy(false)[0]);}
        }
        return out;
}

template<class T>
boost::python::list trainData<T>::transferWeightListToNumpy(){
    boost::python::list out;
    for( auto& a: weight_arrays_){
        out.append(a.transferToNumpy()[0]);
    }
    return out;
}


#endif


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
