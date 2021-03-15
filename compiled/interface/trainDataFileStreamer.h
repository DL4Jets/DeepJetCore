/*
 * trainDataFileStreamer.h
 *
 *  Created on: 15 Mar 2021
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAFILESTREAMER_H_
#define DEEPJETCORE_COMPILED_INTERFACE_TRAINDATAFILESTREAMER_H_

#include "simpleArray.h"
#include <string>

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

class trainDataFileStreamer;

class simpleArrayStreamer{
    friend class trainDataFileStreamer;
public:

    enum dataUsage {feature_data, truth_data, weight_data};

    ~simpleArrayStreamer(){
        clear();
    }

    /**
     * the shape does not include the 'event' dimension
     */
    simpleArrayStreamer(
            const std::string name,
            const std::vector<int>& shape,
            simpleArrayBase::dtypes dtype,
            dataUsage dusage,
            const std::vector<std::string>& featurenames=std::vector<std::string>());

    //maybe replace that with direct 'set' access. TBI
    inline simpleArrayBase & arr(){if(current_) return *current_; else throw std::logic_error("simpleArrayStreamer::arr: no array initialized.");}
    inline const simpleArrayBase & arr()const{if(current_) return *current_; else throw std::logic_error("simpleArrayStreamer::arr: no array initialized.");}

    void fill(){
        arrays_.push_back(current_);
        newCurrentArray();
    }


    simpleArrayBase * copyToFullArray()const;

private:

    void fillEvent();
    void clear();
    void clearData();

    simpleArrayStreamer(){}

    template<class T>
    simpleArrayBase * priv_copyToFullArray()const{
        std::vector<int> newshape = {(int)rowsplits_.size(),-1};//second dimension is the variable one
        //add the actual 'per event' shape
        newshape.insert(newshape.end(), prototype_->shape().begin(),prototype_->shape().end());
        T * outp = new T(newshape,rowsplits_);
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
};


/**
 *
 * Usage:
 * - create the trainDataFileStreamer object (sets output file name and opt. buffer size).
 * - add arrays to it
 * - fill the arrays in the event loop (using arr()->set(i,j,k,l,value) . can have one ragged dimension
 * - finish the event synchronously for all arrays calling fillEvent
 *
 *
 * Example:
 *
 *    trainDataFileStreamer fs("outfile.djctd");
 *    auto features = fs.add("myfeatures",                      // just a name, can also be left blank
 *                           {3},                               // the shape, here just 3 features
 *                           simpleArrayBase::float32,          // the data type
 *                           simpleArrayStreamer::feature_data, // what it's used for
 *                           {"jetpt","jeteta","jetphi"});      // optional feature names
 *
 *    auto truth = fs.add("isSignal",{1},simpleArrayBase::int32,simpleArrayStreamer::truth_data);
 *
 *    for(event: events){
 *
 *        for(jet: jets){
 *            features.arr().set(0, jet->pt());
 *            features.arr().set(1, jet->eta());
 *            features.arr().set(2, jet->phi());
 *            features.fill()
 *        }
 *
 *        truth.arr().set(0, isSUSYevent);
 *        truth.fill()
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
            size_t bufferInMB=20);

    ~trainDataFileStreamer(){
        writeBuffer(true);//write remaining items
        for(auto& a:arraystreamers_a_)
            delete a;
        for(auto& a:arraystreamers_b_)
            delete a;
    }

    template<class T>
    simpleArrayStreamer* add(const std::string name,
            const std::vector<int>& shape,
            simpleArrayBase::dtypes type,
            simpleArrayStreamer::dataUsage dusage,
            const std::vector<std::string>& featurenames=std::vector<std::string>()){
        simpleArrayStreamer* as = new simpleArrayStreamer(name,shape,type,dusage,featurenames);
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


    std::vector<simpleArrayStreamer*> arraystreamers_a_;
    std::vector<simpleArrayStreamer*> arraystreamers_b_;
    std::vector<simpleArrayStreamer*> * activestreamers_;
    std::vector<simpleArrayStreamer*> * writingstreamers_;
    std::string filename_;
    size_t buffermb_;

};


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
