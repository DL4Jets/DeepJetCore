
#include "../interface/trainDataFileStreamer.h"
#include "../interface/trainData.h"

namespace djc{

trainDataFileStreamer::trainDataFileStreamer(
        const std::string & filename,
        float bufferInMB):filename_(filename),buffermb_(bufferInMB){
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
    if(((float)totalsizekb)/1024. >= buffermb_)
        return true;
    return false;
}

namespace test{

#include "helper.h"

std::vector<float> createRandomVector(size_t i){
    return GenerateRandomVector<float>(i);
}

std::vector<std::vector<float> > createRandomVector(size_t i,size_t j){
    std::vector<std::vector<float> > out(i);
    for(size_t ii=0;ii<i;ii++){
        out.at(ii) = GenerateRandomVector<float>(j);
    }
    return out;
}


void testTrainDataFileStreamer(){

    std::string testfilename = "test_testTrainDataFileStreamer_outfile.djctd";

    simpleArray_float32 reference_myfeatures_all;
    reference_myfeatures_all.setFeatureNames({"jetpt","jeteta","jetphi"});
    reference_myfeatures_all.setName("myfeatures");

    simpleArray_float32 reference_myzeropadded_lepton_features_all;
    reference_myzeropadded_lepton_features_all.setFeatureNames({"pt","eta","phi"});
    reference_myzeropadded_lepton_features_all.setName("myzeropadded_lepton_features");
    simpleArray_int32 reference_isSignal_all;
    reference_isSignal_all.setName("isSignal");

    { //file streamer scope
        trainDataFileStreamer fs(testfilename,0.07);//small buffer for testing

        simpleArrayFiller* features = fs.add("myfeatures",                      // just a name, can also be left blank
                {3},                               // the shape, here just 3 features
                simpleArrayBase::float32,          // the data type
                simpleArrayFiller::feature_data, // what it's used for
                true,                              // data is ragged (variable 1st dimension)
                {"jetpt","jeteta","jetphi"});      // optional feature names


        simpleArrayFiller* zeropadded = fs.add("myzeropadded_lepton_features",// just a name, can also be left blank
                {5,3},                             // 3 features each for the first 5 leptons
                simpleArrayBase::float32,          // the data type
                simpleArrayFiller::feature_data, // what it's used for
                false,                             // data is not ragged
                {"pt","eta","phi"});               // optional feature names

        //add a non ragged per-event variable
        simpleArrayFiller* truth = fs.add("isSignal",{1},simpleArrayBase::int32,simpleArrayFiller::truth_data, false);



        for(int i=0;i<3000;i++){
            int nfirst = i+1;
            while(nfirst>30){
                nfirst-=30;
            }
            auto jetprop = createRandomVector(nfirst,3);

            std::vector<int64_t> jetrs={0,nfirst};

            simpleArray_float32 reference_jetarr({1,-1,3},jetrs);
            reference_jetarr.setName("jetarr");

            for(size_t i=0;i<nfirst;i++){
                features->arr().set(0, jetprop[i][0]);
                features->arr().set(1, jetprop[i][1]);
                features->arr().set(2, jetprop[i][2]);
                features->fill();

                reference_jetarr.set(0,i,0, jetprop[i][0]);
                reference_jetarr.set(0,i,1, jetprop[i][1]);
                reference_jetarr.set(0,i,2, jetprop[i][2]);
            }

            //this should not be done excessively in a real setting, needs mem copy every call!
            reference_myfeatures_all.append(reference_jetarr);

            auto lepprop = createRandomVector(5,3);
            zeropadded->arr().fillZeros(); //make sure everything is initialized with zeros

            simpleArray_float32 reference_leparr({1,5,3});
            reference_leparr.setName("leparr");
            reference_leparr.fillZeros();

            for(size_t i=0;i<lepprop.size();i++){
                zeropadded->arr().set(i,0,lepprop.at(i)[0]);
                zeropadded->arr().set(i,1,lepprop.at(i)[1]);
                zeropadded->arr().set(i,2,lepprop.at(i)[2]);

                reference_leparr.set(0,i,0,lepprop.at(i)[0]);
                reference_leparr.set(0,i,1,lepprop.at(i)[1]);
                reference_leparr.set(0,i,2,lepprop.at(i)[2]);
                if(i>3)
                    break;
            }
            zeropadded->fill();

            reference_myzeropadded_lepton_features_all.append(reference_leparr);

            int issignal = i%2;
            truth->arr().set(0, issignal);
            truth->fill();

            simpleArray_int32 reference_issig({1,1});
            reference_issig.setName("issig");
            reference_issig.set(0,0,issignal);
            reference_isSignal_all.append(reference_issig);

            fs.fillEvent();


        }
    }//file streamer scope, auto save

    //read back and check if same
    trainData td;
    td.readFromFile(testfilename);
    td.nFeatureArrays();

    //a bit quick and dirty casts. We know the type (otherwise would need type check first)
    auto reopened_myfeatures = dynamic_cast<simpleArray_float32&>(td.featureArray(0));
    auto reopened_lepton_features = dynamic_cast<simpleArray_float32&>(td.featureArray(1));
    auto reopened_issignal = dynamic_cast<simpleArray_int32&>(td.truthArray(0));

    if(reopened_myfeatures != reference_myfeatures_all){
       // tdfeat0.cout();myfeatures_all.cout();
        throw std::logic_error("testTrainDataFileStreamer: simpleArray_float32 ragged inconsistent");
    }
    if(reopened_lepton_features != reference_myzeropadded_lepton_features_all){
       // tdfeat1.cout();
        throw std::logic_error("testTrainDataFileStreamer: simpleArray_float32 not ragged inconsistent");
    }
    if(reopened_issignal != reference_isSignal_all){
       // tdtruth.cout();
        throw std::logic_error("testTrainDataFileStreamer: simpleArray_float32 ragged inconsistent");
    }
    //avoid warning
    int res = system(("rm -f "+testfilename).data());

}

}//test

}//djc
