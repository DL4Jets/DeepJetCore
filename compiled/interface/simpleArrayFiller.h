/*
 * simpleArrayFiller.h
 *
 *  Created on: 16 Mar 2021
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAYFILLER_H_
#define DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAYFILLER_H_

#include "simpleArray.h"
#include <string>

namespace djc{

class trainDataFileStreamer;
class simpleArrayFiller{
    friend class trainDataFileStreamer;
public:

    enum dataUsage {feature_data, truth_data, weight_data};

    ~simpleArrayFiller(){
        clear();
    }

    /**
     * the shape does not include the 'event' dimension
     */
    simpleArrayFiller(
            const std::string name,
            const std::vector<int>& shape,
            simpleArrayBase::dtypes dtype,
            dataUsage dusage,
            bool isragged,
            const std::vector<std::string>& featurenames=std::vector<std::string>());

    //maybe replace that with direct 'set' access. TBI
    inline simpleArrayBase & arr(){if(current_) return *current_; else throw std::logic_error("simpleArrayStreamer::arr: no array initialized.");}
    inline const simpleArrayBase & arr()const{if(current_) return *current_; else throw std::logic_error("simpleArrayStreamer::arr: no array initialized.");}

    void fill(){
        arrays_.push_back(current_);
        newCurrentArray();
    }


    simpleArrayBase * copyToFullArray()const;

    //TBI
    // tensor moveToTFTensor();

private:

    void fillEvent();
    void clear();
    void clearData();

    simpleArrayFiller(){}

    template<class T>
    simpleArrayBase * priv_copyToFullArray()const{
        std::vector<int> newshape;
        if(isragged_)
            newshape = {(int)rowsplits_.size()-1,-1}; //second dimension is the variable one
        else
            newshape = {(int)arrays_.size()};
        //add the actual 'per event' shape
        newshape.insert(newshape.end(), prototype_->shape().begin(),prototype_->shape().end());
        T * outp = 0;
        if(isragged_)
            outp = new T(newshape,rowsplits_);
        else
            outp = new T(newshape);
        outp->setName(prototype_->name());
        outp->setFeatureNames(prototype_->featureNames());
        size_t counter=0;
        for(const auto& a:arrays_){
            for(size_t i=0;i<a->size();i++){
                outp->data()[counter] = dynamic_cast<T*>(a)->data()[i];
                counter++;
            }
        }
        return outp;
    }

    //this is not exact but good enough for approx buffering
    size_t memSizeKB()const;

    void newCurrentArray();

    //needs to be pointers because of types
    std::vector<simpleArrayBase* > arrays_;
    std::vector<int64_t> rowsplits_;
    simpleArrayBase* current_;
    simpleArrayBase* prototype_;
    dataUsage dusage_;
    bool isragged_;
};

}//djc



#endif /* DEEPJETCORE_COMPILED_INTERFACE_SIMPLEARRAYFILLER_H_ */
