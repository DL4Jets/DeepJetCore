/*
 * trainDataFileStreamer.h
 *
 *  Created on: 15 Mar 2021
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAFILESTREAMER_H_
#define DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAFILESTREAMER_H_

#include "simpleArray.h"
#include "simpleArrayFiller.h"
#include <string>
#include <initializer_list>
/*
 * general idea: just manage access to simpleArrays.
 * Once buffer full, create trainData and write out
 * (make sure this can be threaded)
 *
 *
 */

//helper, might be moved
/*
 * vector of simplearray
 * vector of row splits created when calling next
 *
 * return simpleArrayBase (typeless)
 */
namespace djc{



/**
 *
 * Usage:
 * - create the trainDataFileStreamer object (sets output file name and opt. buffer size).
 * - add arrays to it
 * - fill the arrays in the event loop (using arr()->set(i,j,k,l,value) . can have one ragged dimension
 * - finish the event synchronously for all arrays calling fillEvent
 *
 *
 * Example pseudo code:
 *
 *
 *
 *    trainDataFileStreamer fs("outfile.djctd");
 *    auto features = fs.add("myfeatures",                      // just a name, can also be left blank
 *                           {3},                               // the shape, here just 3 features
 *                           simpleArrayBase::float32,          // the data type
 *                           simpleArrayFiller::feature_data, // what it's used for
 *                           true,                              // data is ragged (variable 1st dimension)
 *                           {"jetpt","jeteta","jetphi"});      // optional feature names
 *
 *
 *    auto zeropadded = fs.add("myzeropadded_lepton_features",// just a name, can also be left blank
 *                           {5,3},                             // 3 features each for the first 5 leptons
 *                           simpleArrayBase::float32,          // the data type
 *                           simpleArrayFiller::feature_data, // what it's used for
 *                           false,                             // data is not ragged
 *                           {"pt","eta","phi"});               // optional feature names
 *
 *    //add a non ragged per-event variable
 *    auto truth = fs.add("isSignal",{1},simpleArrayBase::int32,simpleArrayFiller::truth_data, false);
 *
 *    for(event: events){
 *
 *        for(jet: jets){
 *            features->arr().set(0, jet->pt());
 *            features->arr().set(1, jet->eta());
 *            features->arr().set(2, jet->phi());
 *            features->fill()
 *        }
 *
 *
 *        zeropadded->arr().fillZero(); //make sure everything is initialized with zeros
 *        for(size_t i=0;i<leptons.size();i++){
 *            zeropadded->arr().set(i,0,leptons.at(i).pt());
 *            zeropadded->arr().set(i,1,leptons.at(i).eta());
 *            zeropadded->arr().set(i,2,leptons.at(i).phi());
 *            if(i>3)
 *               break;
 *        }
 *        zeropadded->fill();
 *
 *        truth->arr().set(0, isSUSYevent);
 *        truth->fill()
 *
 *        fs.fillEvent();
 *    }
 *
 *    //no need to explicitly write out/close etc. action is implemented in the destructor
 *
 */
class trainDataFileStreamer {
public:

    trainDataFileStreamer(
            const std::string & filename,
            float bufferInMB=20);

    ~trainDataFileStreamer(){
        writeBuffer(true);//write remaining items
        for(auto& a:arraystreamers_a_)
            delete a;
        for(auto& a:arraystreamers_b_)
            delete a;
    }

    simpleArrayFiller* add(const std::string& name,
            const std::vector<int>& shape,
            simpleArrayBase::dtypes type,
            simpleArrayFiller::dataUsage dusage,
            bool isragged,
            const std::vector<std::string>& featurenames=std::vector<std::string>()){
        simpleArrayFiller* as = new simpleArrayFiller(name,shape,type,dusage,isragged,featurenames);
        activestreamers_->push_back(as);
        return as;
    }



    inline void fillEvent(){
        //makes sure it's in sync
        for(auto& a:*activestreamers_)
            a->fillEvent();
        if(bufferFull())
            writeBuffer();
    }


private:

    void writeBuffer(bool sync=false);
    bool bufferFull();


    std::vector<simpleArrayFiller*> arraystreamers_a_;
    std::vector<simpleArrayFiller*> arraystreamers_b_;
    std::vector<simpleArrayFiller*> * activestreamers_;
    std::vector<simpleArrayFiller*> * writingstreamers_;
    std::string filename_;
    float buffermb_;

};


namespace test{

void testTrainDataFileStreamer();

}


/*
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */


}//djc

#endif /* DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAFILESTREAMER_H_ */
