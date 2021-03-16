
#include "../interface/trainDataFileStreamer.h"
#include "../interface/trainData.h"

namespace djc{

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
        if(a->dusage_ == simpleArrayFiller::feature_data)
            td.storeFeatureArray(*acp);
        else if(a->dusage_ == simpleArrayFiller::truth_data)
            td.storeTruthArray(*acp);
        else if(a->dusage_ == simpleArrayFiller::feature_data)
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
