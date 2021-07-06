/*
 * trainDataGenerator.h
 *
 *  Created on: 7 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_

#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include <boost/python/exception_translator.hpp>
#include "helper.h"
#include "pythonToSTL.h"

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

class trainDataGenerator{
public:
    trainDataGenerator();
    ~trainDataGenerator();

    /**
     * Also opens all files (verify) and gets the total sample size
     */
    void setFileList(const std::vector<std::string>& files){
        clear();
        orig_infiles_=files;
        readInfo();
    }


    void setFileListPy(boost::python::list files);

    void setBuffer(const trainData&);

    void setBatchSize(size_t nelements){
        batchsize_= nelements;
        if(orig_rowsplits_.size())
            prepareSplitting();
    }
    void setSquaredElementsLimit(bool use_sq_limit){
        sqelementslimit_=use_sq_limit;
        if(orig_rowsplits_.size())
            prepareSplitting();
    }
    void setSkipTooLargeBatches(bool skipthem){
        skiplargebatches_=skipthem;
        if(orig_rowsplits_.size())
            prepareSplitting();
    }

    int getNTotal()const{return ntotal_;}

    void setFileTimeout(size_t seconds){
        filetimeout_=seconds;
    }

    int getNBatches()const{return nbatches_;}

    bool lastBatch()const;

    bool isEmpty()const;

    void prepareNextEpoch();

    void shuffleFileList();

    void end();
    /**
     * clears all dataset related info but keeps batch size, file timout etc
     */
    void clear();

    /**
     * gets Batch. If batchsize is specified, it is up to the user
     * to make sure that the sum of all batches is smaller or equal the
     * total sample size.
     * The batch size is always the size of the NEXT batch!
     *
     */
    trainData getBatch(); //if no threading batch index can be given? just for future?

    int debuglevel;



private:
    void readBuffer();
    void readInfo();
    std::vector<int64_t> subShuffleRowSplits(const std::vector<int64_t>& thisrs,
            const std::vector<size_t>& s_idx)const;
    void prepareSplitting();
    bool tdHasRaggedDimension(const trainData& )const;

    trainData  prepareBatch();
    std::vector<std::string> orig_infiles_;
    std::vector<size_t> shuffle_indices_;
    std::vector<std::vector<size_t> > sub_shuffle_indices_;
    std::vector<std::vector<int64_t> > orig_rowsplits_;
    std::vector<size_t> splits_;
    std::vector<bool> usebatch_;
    int randomcount_;
    size_t batchsize_;
    bool sqelementslimit_,skiplargebatches_;

    trainData buffer_store, buffer_read;
    std::thread * readthread_;
    std::string nextread_;
    size_t nextreadIdx_;
    size_t filecount_;
    size_t nbatches_;
    size_t npossiblebatches_;
    size_t ntotal_;
    size_t nsamplesprocessed_;
    size_t lastbatchsize_;
    size_t filetimeout_;
    size_t batchcount_;
    size_t lastbuffersplit_;
};


}//namespace
#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAGENERATOR_H_ */
