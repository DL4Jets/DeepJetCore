/*
 * trainDataInterface.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_
#define DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAINTERFACE_H_

//#define DJC_DATASTRUCTURE_PYTHON_BINDINGS//DEBUG

#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include "helper.h"

#include "simpleArray.h"
#include <stdio.h>
#include "IO.h"

#include <iostream>

namespace djc{

/*
 * use small helper class to store simpleArrayBase pointers
 * and manage ownership where needed.
 * just wrap around std::vector
 */
class typeContainer{
public:

    void push_back(simpleArrayBase& a);
    void move_back(simpleArrayBase& a);

    bool operator==(const typeContainer& rhs)const;
    bool operator!=(const typeContainer& rhs)const{
        return !(*this==rhs);
    }


    simpleArrayBase& at(size_t idx);
    const simpleArrayBase& at(size_t idx)const;

    simpleArrayBase::dtypes dtype(size_t idx)const{return at(idx).dtype();}

    simpleArray_float32& at_asfloat32(size_t idx);
    const simpleArray_float32& at_asfloat32(size_t idx)const;
    simpleArray_int32& at_asint32(size_t idx);
    const simpleArray_int32& at_asint32(size_t idx)const;

    void clear();

    size_t size()const{return sorting_.size();}


    void writeToFile(FILE *&) const;
    inline void readFromFile(FILE *&f){
        readFromFile_priv(f,false);
    }

    inline void readMetaDataFromFile(FILE *&f){//produces size 0 arrays with correct dtypes and shapes otherwise
        readFromFile_priv(f,true);
    }

private:
    void readFromFile_priv(FILE *& f, bool justmetadata);

    std::vector<simpleArray_float32> farrs_;
    std::vector<simpleArray_int32> iarrs_;

    enum typesorting{isfloat,isint};
    std::vector<std::pair<typesorting,size_t> > sorting_;

};


/*
 * The idea is to make this a fixed size array class, that is filled with data and then written out once full.
 * a truncate function will allow to  truncate arrays at a given position.
 * This is memory intense, but can be written out in small pieces and then merged
 *
 * No checks on the first dimension because of possibly ragged arrays
 */

class trainData{
public:



    bool operator==(const trainData& rhs)const;
    bool operator!=(const trainData& rhs)const{
        return !(*this==rhs);
    }
    //takes ownership
    //these need to be separated by input type because python does not allow for overload
    //but then the py interface can be made generic  to accept differnt types

    //make this a base reference and then check for dtype and cast
    //
    int storeFeatureArray( simpleArrayBase&);
    int storeTruthArray( simpleArrayBase&);
    int storeWeightArray( simpleArrayBase&);

    //for python, no implicit cast
    inline int storeFeatureArray( simpleArray_float32& a){
        return storeFeatureArray(dynamic_cast<simpleArrayBase&> (a));
    }
    inline int storeTruthArray( simpleArray_float32& a){
        return storeTruthArray(dynamic_cast<simpleArrayBase&> (a));
    }
    inline int storeWeightArray( simpleArray_float32& a){
        return storeWeightArray(dynamic_cast<simpleArrayBase&> (a));
    }

    inline int storeFeatureArray( simpleArray_int32&a){
        return storeFeatureArray(dynamic_cast<simpleArrayBase&> (a));
    }
    inline int storeTruthArray( simpleArray_int32& a){
        return storeTruthArray(dynamic_cast<simpleArrayBase&> (a));
    }
    inline int storeWeightArray( simpleArray_int32& a){
        return storeWeightArray(dynamic_cast<simpleArrayBase&> (a));
    }

    //these are not really used so much -->
    /*
     * This class actually doesn't really need data operations. so it can implement
     * only simpleArrayBase calls
     *
     *
     */

    const simpleArrayBase & featureArray(size_t idx) const {
        return feature_arrays_.at(idx);
    }

    const simpleArrayBase & truthArray(size_t idx) const {
        return truth_arrays_.at(idx);
    }

    const simpleArrayBase & weightArray(size_t idx) const {
        return weight_arrays_.at(idx);
    }

    simpleArrayBase & featureArray(size_t idx)  {
        return feature_arrays_.at(idx);
    }

    simpleArrayBase & truthArray(size_t idx)  {
        return truth_arrays_.at(idx);
    }

    simpleArrayBase & weightArray(size_t idx)  {
        return weight_arrays_.at(idx);
    }

    //<---

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
    void append(const trainData& );

    /*
     * split along first axis
     * Returns the second part, leaves the first.
     */
    trainData split(size_t splitindex);
    trainData getSlice(size_t splitindex_begin, size_t splitindex_end)const;

    trainData shuffle(const std::vector<size_t>& shuffle_idxs)const;

    bool validSlice(size_t splitindex_begin, size_t splitindex_end)const ;

    /*
     *
     */
    size_t nElements()const{
        if(feature_shapes_.size() && feature_shapes_.at(0).size())
            return feature_shapes_.at(0).at(0);
        else
            return 0;
    }

    int nTotalElements()const{
        if(feature_shapes_.size() && feature_shapes_.at(0).size()){
            int ntotalelems=0;
            for(size_t i=0;i< feature_shapes_.at(0).size(); i++){
                ntotalelems = feature_shapes_.at(0).at(i);
                if(i>0 && ntotalelems<0)
                    return std::abs(ntotalelems);
                else if(i>0)
                    return feature_shapes_.at(0).at(0);
            }
        }
        else
            return 0;
        return 0;
    }

    const std::vector<std::vector<int> > & featureShapes()const{return  feature_shapes_;}
    const std::vector<std::vector<int> > & truthShapes()const{return  truth_shapes_;}
    const std::vector<std::vector<int> > & weightShapes()const{return  weight_shapes_;}

    void writeToFile(std::string filename)const;
    void addToFile(std::string filename)const;

    void addToFileP(FILE *& f)const;

    void readFromFile(std::string filename){
        priv_readFromFile(filename,false);
    }
    void readFromFileBuffered(std::string filename){
        priv_readFromFile(filename,true);
    }

    //could use a readshape or something!
    void readMetaDataFromFile(const std::string& filename);

    std::vector<int64_t> getFirstRowsplits()const;
    std::vector<int64_t> readShapesAndRowSplitsFromFile(const std::string& filename, bool checkConsistency=true);

    void clear();

    trainData copy()const {return *this;}
    //from python
    void skim(size_t batchelement);



    inline boost::python::list getNumpyFeatureShapes()const{
        return transferShapesToPyList(feature_shapes_);
    }
    inline boost::python::list getNumpyTruthShapes()const{
        return transferShapesToPyList(truth_shapes_);
    }
    inline boost::python::list getNumpyWeightShapes()const{
        return transferShapesToPyList(weight_shapes_);
    }

    inline boost::python::list getNumpyFeatureDTypes()const{
        return transferDTypesToPyList(feature_arrays_);
    }
    inline boost::python::list getNumpyTruthDTypes()const{
        return transferDTypesToPyList(truth_arrays_);
    }
    inline boost::python::list getNumpyWeightDTypes()const{
        return transferDTypesToPyList(weight_arrays_);
    }

    inline boost::python::list getNumpyFeatureArrayNames()const{
        return transferNamesToPyList(feature_arrays_);
    }
    inline boost::python::list getNumpyTruthArrayNames()const{
        return transferNamesToPyList(truth_arrays_);
    }
    inline boost::python::list getNumpyWeightArrayNames()const{
        return transferNamesToPyList(weight_arrays_);
    }

    //has ragged support
    boost::python::list transferFeatureListToNumpy(bool padrowsplits=false);

    //has ragged support
    boost::python::list transferTruthListToNumpy(bool padrowsplits=false);

    //no ragged support
    boost::python::list transferWeightListToNumpy(bool padrowsplits=false);


    boost::python::list getTruthRaggedFlags()const;

    /*
     * the following ones can be improved w.r.t. performance
     */


    //has ragged support
    boost::python::list copyFeatureListToNumpy(bool padrowsplits=false){
        auto td = *this;
        return td.transferFeatureListToNumpy(padrowsplits); //fast hack
    }

    //has ragged support
    boost::python::list copyTruthListToNumpy(bool padrowsplits=false){
        auto td = *this;
        return td.transferTruthListToNumpy(padrowsplits); //fast hack
    }

    //no ragged support
    boost::python::list copyWeightListToNumpy(bool padrowsplits=false){
        auto td = *this;
        return td.transferWeightListToNumpy(padrowsplits); //fast hack
    }


private:

    void priv_readFromFile(std::string filename, bool memcp);

    trainData priv_readFromFileP(FILE *& f, const std::string& filename)const;
    void priv_readSelfFromFileP(FILE *& f, const std::string& filename);

    void checkFile(FILE *& f, const std::string& filename="")const;


    void readRowSplitArray(FILE *&, std::vector<int64_t> &rs, bool check)const;

    std::vector<std::vector<int> > getShapes(const typeContainer& a)const;

    template <class U>
    void writeNested(const std::vector<std::vector<U> >& v, FILE *&)const;
    template <class U>
    void readNested( std::vector<std::vector<U> >& v, FILE *&)const;

    void updateShapes();

    boost::python::list transferNamesToPyList(const typeContainer&)const;
    boost::python::list transferShapesToPyList(const std::vector<std::vector<int> >&)const;
    boost::python::list transferDTypesToPyList(const typeContainer&)const;


    typeContainer feature_arrays_;
    typeContainer truth_arrays_;
    typeContainer weight_arrays_;

    std::vector<std::vector<int> > feature_shapes_;
    std::vector<std::vector<int> > truth_shapes_;
    std::vector<std::vector<int> > weight_shapes_;


    boost::python::list transferToNumpyList(typeContainer& , bool pad_rowsplits);


};




/*
 * append along first axis
 */

/*
 * split along first axis
 * Returns the first part, leaves the second.
 *
 * Can use some performance improvements
 */











template <class U>
void trainData::writeNested(const std::vector<std::vector<U> >& v, FILE *& ofile)const{

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

template <class U>
void trainData::readNested(std::vector<std::vector<U> >& v, FILE *& ifile)const{

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
