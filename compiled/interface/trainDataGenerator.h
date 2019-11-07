/*
 * trainDataGenerator.h
 *
 *  Created on: 7 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_

#include <string>
#include <vector>
#include "trainData.h"
#include <algorithm>
#include <random>
#include <iterator>
#include <thread>

namespace djc{

/*
 * Base class, no numpy interface or anything yet.
 * Inherit from/use this class and define the actual batch feed function.
 * This could as well be filling a (ragged) tensorflow tensor
 */
template <class T>
class trainDataGenerator{
public:
    trainDataGenerator();
    ~trainDataGenerator();

    void setFileList(const std::vector<std::string>& files){
        orig_infiles_=files;
        shuffled_infiles_=orig_infiles_;
    }
    void setBatchSize(size_t nelements){
        batchsize_=nelements;
        nbatches_ = ntotal_/batchsize_;
    }
    void setNTotal(size_t n){
        ntotal_=n;
        nbatches_ = ntotal_/batchsize_;
    }

    size_t getNBatches()const{
        return nbatches_;
    }

    void beginEpoch();
    void endEpoch();

    /**
     * gets Batch. If batchsize is specified, it is up to the user
     * to make sure that the sum of all batches is smaller or equal the
     * total sample size
     */
    trainData<T> getBatch(size_t batchsize=0);

private:
    void shuffleFilelist();
    void readBuffer();
    std::vector<std::string> orig_infiles_;
    std::vector<std::string> shuffled_infiles_;
    int randomcount_;
    size_t batchsize_;

    trainData<T> buffer_store, buffer_read;
    std::thread * readthread_;
    std::string nextread_;
    size_t filecount_;
    size_t nbatches_;
    size_t ntotal_;
    size_t nprocessed_;
};

template<class T>
trainDataGenerator<T>::trainDataGenerator() :
        randomcount_(1), batchsize_(2), readthread_(0), filecount_(0), nbatches_(
                0), ntotal_(0), nprocessed_(0) {
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
    std::shuffle(std::begin(shuffled_infiles_),std::end(shuffled_infiles_),g);
}



template<class T>
void trainDataGenerator<T>::readBuffer(){
    buffer_read.readFromFile(nextread_);
}




template<class T>
void trainDataGenerator<T>::beginEpoch(){
    if(!readthread_){//no pre-read going on
        nprocessed_=0;
        filecount_=0;
        buffer_store.clear();
        buffer_read.clear();
    }
}
template<class T>
void trainDataGenerator<T>::endEpoch(){

    //prepare for next epoch, pre-read first file
    if(readthread_){
        readthread_->join();
        delete readthread_;
    }
    buffer_store.clear();
    buffer_read.clear();

    shuffleFilelist();
    filecount_=0;
    nprocessed_=0;
    nextread_ = shuffled_infiles_.at(filecount_);
    readthread_ = new std::thread(&trainDataGenerator<T>::readBuffer,this);
}

template<class T>
trainData<T> trainDataGenerator<T>::getBatch(size_t batchsize){

    size_t bufferelements=buffer_store.nElements();

    if(!batchsize)
        batchsize=batchsize_;

    while(bufferelements<batchsize){
        //if thread, read join
        if(readthread_){
            readthread_->join();
            delete readthread_;
            readthread_=0;
        }
        buffer_store.append(buffer_read);
        buffer_read.clear();
        bufferelements = buffer_store.nElements();

        if(nprocessed_ + bufferelements < ntotal_){
            if (filecount_ >= shuffled_infiles_.size())
                throw std::runtime_error(
                        "trainDataGenerator<T>::getBatch: more batches requested than data in the sample");

            nextread_ = shuffled_infiles_.at(filecount_);
            filecount_++;
            readthread_ = new std::thread(&trainDataGenerator<T>::readBuffer,this);
        }
    }
    nprocessed_+=batchsize_;
    return buffer_store.split(batchsize);
}



}//namespace
#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_ */
