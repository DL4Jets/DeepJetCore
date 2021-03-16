
#include "../interface/simpleArrayFiller.h"

namespace djc{


simpleArrayFiller::simpleArrayFiller(
        const std::string name,
        const std::vector<int>& shape,
        simpleArrayBase::dtypes type,
        dataUsage dusage,
        bool isragged,
        const std::vector<std::string>& featurenames){

    dusage_=dusage;
    isragged_=isragged;

    if(type==simpleArrayBase::float32)
        prototype_= new simpleArray_float32(shape);
    else if(type==simpleArrayBase::int32)
        prototype_= new simpleArray_int32(shape);
    else
        throw std::invalid_argument("simpleArrayStreamer::init: unsupported dtype");

    prototype_->setFeatureNames(featurenames);
    prototype_->setName(name);

    current_=0;
    if(isragged_)
        rowsplits_.push_back(0);
    newCurrentArray();

}

void simpleArrayFiller::clear(){
    clearData();
    if(prototype_)
        delete prototype_;
    prototype_=0;
}

void simpleArrayFiller::clearData(){//keep prototype
    for(auto& a:arrays_)
        delete a;
    arrays_.clear();
    rowsplits_.clear();
    if(isragged_)
        rowsplits_.push_back(0);
    if(current_)
        delete current_;
    current_=0;
    newCurrentArray();
}


void simpleArrayFiller::fillEvent(){
    rowsplits_.push_back(arrays_.size());
}


simpleArrayBase * simpleArrayFiller::copyToFullArray()const{
    if(prototype_->dtype() == simpleArrayBase::float32){
       return priv_copyToFullArray<simpleArray_float32>();
    }
    else if(prototype_->dtype() == simpleArrayBase::int32){
        return priv_copyToFullArray<simpleArray_int32>();
    }
    else
        throw std::runtime_error("simpleArrayStreamer::copyToFullArray: unrecognised type");
    return 0;
}

size_t simpleArrayFiller::memSizeKB()const{
    size_t vsize = arrays_.size();
    size_t pts = prototype_->size();
    size_t datasize = vsize*pts + pts;

    datasize *= sizeof(float); //int32 and float32 both 4 bytes

    size_t rssize = rowsplits_.size() * sizeof(int64_t);

    return (datasize + rssize)/1024;
}

void simpleArrayFiller::newCurrentArray(){
    if(prototype_->dtype() == simpleArrayBase::float32)
        current_ = new simpleArray_float32((*(simpleArray_float32*)prototype_));
    else if(prototype_->dtype() == simpleArrayBase::int32)
        current_ = new simpleArray_int32((*(simpleArray_int32*)prototype_));
}

}//djc


