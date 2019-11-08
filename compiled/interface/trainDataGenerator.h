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
#include <iostream>

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

    /**
     * Also opens all files (verify) and gets the total sample size
     */
    void setFileList(const std::vector<std::string>& files){
        orig_infiles_=files;
        shuffled_infiles_=orig_infiles_;
        readNTotal();
    }
    void setBatchSize(size_t nelements){
        batchsize_=nelements;
        nbatches_ = ntotal_/batchsize_;
    }
    size_t getNTotal()const{return ntotal_;}

    void setFileTimeout(size_t seconds){
        filetimeout_=seconds;
    }

    size_t getNBatches()const{
        return nbatches_;
    }

    bool lastBatch()const{
        return nsamplesprocessed_ >= ntotal_ - lastbatchsize_;
    }

    void prepareNextEpoch();

    /**
     * gets Batch. If batchsize is specified, it is up to the user
     * to make sure that the sum of all batches is smaller or equal the
     * total sample size
     */
    trainData<T> getBatch(size_t batchsize=0);

    bool debug;
private:
    void shuffleFilelist();
    void readBuffer();
    void readNTotal();
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
    size_t nsamplesprocessed_;
    size_t lastbatchsize_;
    size_t filetimeout_;
};


template<class T>
trainDataGenerator<T>::trainDataGenerator() :debug(false),
        randomcount_(1), batchsize_(2), readthread_(0), filecount_(0), nbatches_(
                0), ntotal_(0), nsamplesprocessed_(0),filetimeout_(10) {
}

template<class T>
trainDataGenerator<T>::~trainDataGenerator(){
    if(readthread_){
        //try{
        //    delete readthread_;
        //}catch(...){}
        //buffer_read.clear();
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
    size_t ntries = 0;
    while(ntries < filetimeout_){
        if(io::fileExists(nextread_)){
            buffer_read.readFromFile(nextread_);
            return;
        }
        sleep(1);
    }
    throw std::runtime_error("trainDataGenerator<T>::readBuffer: file "+nextread_+ " could not be read.");
}


template<class T>
void trainDataGenerator<T>::readNTotal(){
    ntotal_=0;
    for(const auto& f: orig_infiles_){
        trainData<T> td;
        std::vector<std::vector<int> > fs, ts, ws;
        td.readShapesFromFile(f,fs, ts, ws);
        //first dimension is always Nelements. At least features are filled
        if(fs.size()<1 || fs.at(0).size()<1)
            throw std::runtime_error("trainDataGenerator<T>::readNTotal: no features filled in trainData object");
        ntotal_ += fs.at(0).at(0);
    }
    nbatches_ = ntotal_/batchsize_;
}


template<class T>
void trainDataGenerator<T>::prepareNextEpoch(){

    //prepare for next epoch, pre-read first file
    if(readthread_){
        readthread_->join(); //this is slow! FIXME: better way to exit gracefully in a simple way
        delete readthread_;

        //try{ //brute force but might be ok
        //    delete readthread_;
        //}catch(...){}
        //buffer_read.clear();
    }
    buffer_store.clear();
    buffer_read.clear();
    filecount_=0;
    nsamplesprocessed_=0;

    shuffleFilelist();
    nextread_ = shuffled_infiles_.at(filecount_);
    readthread_ = new std::thread(&trainDataGenerator<T>::readBuffer,this);
}

template<class T>
trainData<T> trainDataGenerator<T>::getBatch(size_t batchsize){

    size_t bufferelements=buffer_store.nElements();

    if(!batchsize)
        batchsize=batchsize_;
    lastbatchsize_=batchsize;
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

        if(debug)
            std::cout << "nprocessed " << nsamplesprocessed_ << " file " << filecount_ << " in buffer " << bufferelements
            << " file read " << nextread_ << " totalfiles " << shuffled_infiles_.size() << std::endl;

        if(nsamplesprocessed_ + bufferelements < ntotal_){
            if (filecount_ >= shuffled_infiles_.size())
                throw std::runtime_error(
                        "trainDataGenerator<T>::getBatch: more batches requested than data in the sample");

            nextread_ = shuffled_infiles_.at(filecount_);
            filecount_++;
            readthread_ = new std::thread(&trainDataGenerator<T>::readBuffer,this);
        }
    }
    nsamplesprocessed_+=batchsize;
    return buffer_store.split(batchsize);
}



}//namespace
#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_ */
