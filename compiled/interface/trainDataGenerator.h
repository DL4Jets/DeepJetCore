/*
 * trainDataGenerator.h
 *
 *  Created on: 7 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_

#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include "helper.h"
#include "pythonToSTL.h"
#endif

#include <string>
#include <vector>
#include "trainData.h"
#include <algorithm>
#include <random>
#include <iterator>
#include <thread>
#include <iostream>

namespace djc{

/*
 * Base class, no numpy interface or anything yet.
 * Inherit from/use this class and define the actual batch feed function.
 * This could as well be filling a (ragged) tensorflow tensor
 *
 *
 * Notes for future improvements:
 *
 *  - pre-split trainData in buffer (just make it a vector/fifo-like queue)
 *    propagates to trainData, simpleArray, then make multiple memcpy (even threaded?)
 *    (but not read - it is still a split!)
 *    This makes the second thread obsolete, and still everything way faster!
 *
 *  - for ragged: instead of batch size, set upper limit on data size (number of floats)
 *    can be used to pre-split in a similar way
 *
 *
 *
 *
 */
template <class T>
class trainDataGenerator{
public:
    trainDataGenerator();
    ~trainDataGenerator();

    /**
     * Also opens all files (verify) and gets the total sample size
     */
    void setFileList(const std::vector<std::string>& files){
        orig_infiles_=files;
        shuffle_indices_.resize(orig_infiles_.size());
        for(size_t i=0;i<shuffle_indices_.size();i++)
            shuffle_indices_[i]=i;
        readInfo();
    }
    void setBatchSize(size_t nelements){
        batchsize_= nelements;
        if(orig_rowsplits_.size())
            prepareSplitting();
        nbatches_ = getNBatches();
    }
    int getNTotal()const{return ntotal_;}

    void setFileTimeout(size_t seconds){
        filetimeout_=seconds;
    }

    int getNBatches()const;

    bool lastBatch()const;

    void prepareNextEpoch();

    void shuffleFilelist();

    void end();


    /**
     * gets Batch. If batchsize is specified, it is up to the user
     * to make sure that the sum of all batches is smaller or equal the
     * total sample size.
     * The batch size is always the size of the NEXT batch!
     *
     */
    trainData<T> getBatch(); //if no threading batch index can be given? just for future?

    bool debug;

#ifdef DJC_DATASTRUCTURE_PYTHON_BINDINGS
    void setFileListP(boost::python::list files){
        djc::trainDataGenerator<T>::setFileList(toSTLVector<std::string>(files));
    }
#endif


private:
    void readBuffer();
    void readInfo();
    void prepareSplitting();

    trainData<T>  prepareBatch();
    std::vector<std::string> orig_infiles_;
    std::vector<size_t> shuffle_indices_;
    std::vector<std::vector<size_t> > orig_rowsplits_;
    std::vector<size_t> splits_;
    int randomcount_;
    size_t batchsize_;

    trainData<T> buffer_store, buffer_read;
    std::thread * readthread_;
    std::string nextread_;
    size_t filecount_;
    size_t nbatches_;
    size_t ntotal_;
    size_t nsamplesprocessed_;
    size_t lastbatchsize_;
    size_t filetimeout_;
    size_t batchcount_;

};


template<class T>
trainDataGenerator<T>::trainDataGenerator() :debug(false),
        randomcount_(1), batchsize_(2), readthread_(0), filecount_(0), nbatches_(
                0), ntotal_(0), nsamplesprocessed_(0),lastbatchsize_(0),filetimeout_(10),
                batchcount_(0){
}

template<class T>
trainDataGenerator<T>::~trainDataGenerator(){
    if(readthread_){
        readthread_->join();
        delete readthread_;
    }

}

template<class T>
void trainDataGenerator<T>::shuffleFilelist(){
    std::random_device rd;
    std::mt19937 g(rd());
    g.seed(randomcount_);
    randomcount_++;
    std::shuffle(std::begin(shuffle_indices_),std::end(shuffle_indices_),g);

    //redo splits etc
    prepareSplitting();
    batchcount_=0;
}



template<class T>
void trainDataGenerator<T>::readBuffer(){
    size_t ntries = 0;
    std::exception caught;
    while(ntries < filetimeout_){
        if(io::fileExists(nextread_)){
            try{
                buffer_read.readFromFile(nextread_);
                return;
            }
            catch(std::exception & e){ //if there are data glitches we don't want the whole training fail immediately
                caught=e;
                std::cout << "File not "<< nextread_ <<" successfully read: " << e.what() << std::endl;
                std::cout << "trying " << filetimeout_-ntries << " more time(s)" << std::endl;
                ntries+=1;
            }
        }
        sleep(1);
        ntries++;
    }
    buffer_read.clear();
    throw std::runtime_error("trainDataGenerator<T>::readBuffer: file "+nextread_+ " could not be read.");
}


template<class T>
void trainDataGenerator<T>::readInfo(){
    ntotal_=0;
    bool hasRagged=false;
    bool firstfile=true;
    for(const auto& f: orig_infiles_){
        trainData<T> td;

        if(! hasRagged || firstfile){
            std::vector<std::vector<int> > fs, ts, ws;
            td.readShapesFromFile(f);
            //first dimension is always Nelements. At least features are filled
            if(td.featureShapes().size()<1 || td.featureShapes().at(0).size()<1)
                throw std::runtime_error("trainDataGenerator<T>::readNTotal: no features filled in trainData object "+f);
        }
        if(firstfile){
            for(const auto& sv: td.featureShapes())
                for(const auto& s:sv)
                    if(s<0)
                        hasRagged=true;
            for(const auto& sv: td.truthShapes())
                for(const auto& s:sv)
                    if(s<0)
                        hasRagged=true;
            for(const auto& sv: td.weightShapes())
                for(const auto& s:sv)
                    if(s<0)
                        hasRagged=true;

        }
        if(hasRagged){
            std::vector<size_t> rowsplits = td.readShapesAndRowSplitsFromFile(f, firstfile);//check consistency only for first
            orig_rowsplits_.push_back(rowsplits);
        }
        firstfile=false;
        ntotal_ += td.nElements();
    }
    batchcount_=0;
    prepareSplitting();
    nbatches_ = getNBatches();
}


template<class T>
void trainDataGenerator<T>::prepareSplitting(){
    splits_.clear();
    if(orig_rowsplits_.size()<1)
        return;
    std::vector<size_t> allrs;
    for(size_t i=0;i<orig_rowsplits_.size();i++){
        const auto& thisrs = orig_rowsplits_.at(shuffle_indices_.at(i));
        if(i==0 || allrs.size()==0){
            allrs=thisrs;}
        else{
            size_t lastelemidx = allrs.size()-1;
            size_t lastnelements = allrs.at(lastelemidx);
            allrs.resize(lastelemidx+thisrs.size());
            for(size_t j=0;j<thisrs.size();j++)
                allrs.at(lastelemidx+j) = lastnelements+thisrs.at(j);
        }
    }

    //DEBUG
    std::cout << "all row splits " <<  allrs.size() << std::endl;
    for(const auto& s: allrs)
        std::cout << s << ", " ;
    std::cout << std::endl;
    return;

    size_t startat=0;
    for(size_t i=0;i<allrs.size();i++){
        auto splitp = simpleArray<T>::findElementSplitPoint(allrs, batchsize_, startat);

        int thisbatch = splitp = startat;

    }
}

template<class T>
int trainDataGenerator<T>::getNBatches()const{
    if(orig_rowsplits_.size()<1)
        return ntotal_/batchsize_;

    return splits_.size();
}


template<class T>
bool trainDataGenerator<T>::lastBatch()const{
    return batchcount_ >= getNBatches() -1 ;
}


template<class T>
void trainDataGenerator<T>::prepareNextEpoch(){

    //prepare for next epoch, pre-read first file
    if(readthread_){
        readthread_->join(); //this is slow! FIXME: better way to exit gracefully in a simple way
        delete readthread_;

    }
    buffer_store.clear();
    buffer_read.clear();
    filecount_=0;
    nsamplesprocessed_=0;
    batchcount_=0;
    nextread_ = orig_infiles_.at(shuffle_indices_.at(filecount_));
    readthread_ = new std::thread(&trainDataGenerator<T>::readBuffer,this);
}
template<class T>
void trainDataGenerator<T>::end(){
    if(readthread_){
        readthread_->join(); //this is slow! FIXME: better way to exit gracefully in a simple way
        delete readthread_;
        readthread_=0;
    }
}

template<class T>
trainData<T> trainDataGenerator<T>::getBatch(){
    return prepareBatch();
}

template<class T>
trainData<T>  trainDataGenerator<T>::prepareBatch(){

    size_t bufferelements=buffer_store.nElements();

    while(bufferelements<batchsize_){
        //if thread, read join
        if(readthread_){
            readthread_->join();
            delete readthread_;
            readthread_=0;
        }
        buffer_store.append(buffer_read);
        buffer_read.clear();
        bufferelements = buffer_store.nElements();

        if(debug)
            std::cout << "nprocessed " << nsamplesprocessed_ << " file " << filecount_ << " in buffer " << bufferelements
            << " file read " << nextread_ << " totalfiles " << orig_infiles_.size() << std::endl;

        if(nsamplesprocessed_ + bufferelements < ntotal_){
            if (filecount_ >= orig_infiles_.size())
                throw std::runtime_error(
                        "trainDataGenerator<T>::getBatch: more batches requested than data in the sample");

            nextread_ = orig_infiles_.at(shuffle_indices_.at(filecount_));
            filecount_++;
            readthread_ = new std::thread(&trainDataGenerator<T>::readBuffer,this);
        }
    }
    if(debug)
        std::cout << "provided batch " << nsamplesprocessed_ << "-" << nsamplesprocessed_+batchsize_ <<
        " elements in buffer: " << bufferelements << std::endl;
    nsamplesprocessed_+=batchsize_;
    lastbatchsize_ = batchsize_;

    trainData<T> out;
    if(splits_.size())
        out = buffer_store.split(splits_.at(batchcount_));
    else
        out = buffer_store.split(batchsize_);

    batchcount_++;
    return out;
}



}//namespace
#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_ */
