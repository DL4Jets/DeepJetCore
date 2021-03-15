
#include "../interface/trainDataFileStreamer.h"
#include "../interface/trainData.h"

namespace djc{


simpleArrayStreamer::simpleArrayStreamer(
        const std::string name,
        const std::vector<int>& shape,
        simpleArrayBase::dtypes type,
        dataUsage dusage,
        const std::vector<std::string>& featurenames){

    dusage_=dusage;

    if(type==simpleArrayBase::float32)
        prototype_= new simpleArray_float32(shape);
    else if(type==simpleArrayBase::int32)
        prototype_= new simpleArray_int32(shape);
    else
        throw std::invalid_argument("simpleArrayStreamer::init: unsupported dtype");

    prototype_->setFeatureNames(featurenames);
    prototype_->setName(name);

    current_=0;
    rowsplits_.push_back(0);
    newCurrentArray();

}

void simpleArrayStreamer::clear(){
    clearData();
    if(prototype_)
        delete prototype_;
    prototype_=0;
}

void simpleArrayStreamer::clearData(){//keep prototype
    for(auto& a:arrays_)
        delete a;
    arrays_.clear();
    rowsplits_.clear();
    if(current_)
        delete current_;
    current_=0;
    newCurrentArray();
}


void simpleArrayStreamer::fillEvent(){
    rowsplits_.push_back(arrays_.size());
}


simpleArrayBase * simpleArrayStreamer::copyToFullArray()const{
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

size_t simpleArrayStreamer::memSizeKB()const{
    size_t vsize = arrays_.size();
    size_t pts = prototype_->size();
    size_t datasize = vsize*pts + pts;

    datasize *= sizeof(float); //int32 and float32 both 4 bytes

    size_t rssize = rowsplits_.size() * sizeof(int64_t);

    return (datasize + rssize)/1024;
}

void simpleArrayStreamer::newCurrentArray(){
    if(prototype_->dtype() == simpleArrayBase::float32)
        current_ = new simpleArray_float32((*(simpleArray_float32*)prototype_));
    else if(prototype_->dtype() == simpleArrayBase::int32)
        current_ = new simpleArray_int32((*(simpleArray_int32*)prototype_));
}


////////// ----- trainDataFileStreamer --- //////////

trainDataFileStreamer::trainDataFileStreamer(
        const std::string & filename,
        size_t bufferInMB):filename_(filename),buffermb_(bufferInMB){
    //create the file
    FILE *ofile = fopen(filename.data(), "wb");
    fclose(ofile);

    activestreamers_=&arraystreamers_a_; //not threaded yet
    writingstreamers_=&arraystreamers_b_;
}


void trainDataFileStreamer::writeBuffer(bool sync){//sync has no effect yet

    auto writestreamers = activestreamers_;//not threaded yet

    trainData td;
    for(auto& a: *writestreamers){
        auto acp = a->copyToFullArray();
        if(a->dusage_ == simpleArrayStreamer::feature_data)
            td.storeFeatureArray(*acp);
        else if(a->dusage_ == simpleArrayStreamer::truth_data)
            td.storeTruthArray(*acp);
        else if(a->dusage_ == simpleArrayStreamer::feature_data)
            td.storeWeightArray(*acp);

        //clean up
        a->clearData();
    }

    td.addToFile(filename_);

}

bool trainDataFileStreamer::bufferFull(){
    size_t totalsizekb=0;
    for(auto& a: *activestreamers_)
        totalsizekb += a->memSizeKB();
    if(totalsizekb/1024 >= buffermb_)
        return true;
    return false;
}



}//djc
