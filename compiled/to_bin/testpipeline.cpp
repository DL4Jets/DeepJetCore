

#include <iostream>
#include "../interface/quicklzWrapper.h"
#include "../interface/simpleArray.h"
#include "../interface/trainData.h"

#include "../interface/trainDataGenerator.h"

std::ostream& operator<< (std::ostream& os, std::vector<int> v){
    for(const auto& i:v)
        os<< i <<" ";
    return os;
}

std::ostream& operator<< (std::ostream& os, std::vector<float> v){
    for(const auto& i:v)
        os<< i <<" ";
    return os;
}

using namespace djc;

int main(){

#ifdef igonorefownow
    bool write=true;
    if(write){
        trainData<float> bigtd;

        auto fidx = bigtd.addFeatureArray({1000, 400, 10});
        for(size_t i=0;i<bigtd.featureArray(fidx).size();i++)
            bigtd.featureArray(fidx).data()[i]=i;

        fidx = bigtd.addTruthArray({1000, 2, 3});
        for(size_t i=0;i<bigtd.truthArray(fidx).size();i++)
            bigtd.truthArray(fidx).data()[i]=i;

        bigtd.writeToFile("bigfile1.djctd");
        bigtd.writeToFile("bigfile2.djctd");
        bigtd.writeToFile("bigfile3.djctd");
        bigtd.writeToFile("bigfile4.djctd");

        bigtd.clear();
        return 1;
    }


    trainData<float> test;
    std::vector<std::vector<int> > fs, ts, ws;
    std::cout << "reading" << std::endl;
    test.readShapesFromFile("bigfile1.djctd",fs,ts,ws);

    std::cout << fs.size() << " " << ts.size() << "  " << ws.size() << std::endl;
    for(const auto& s: fs)
            std::cout << s << std::endl;
    for(const auto& s: ts)
                std::cout << s << std::endl;




    std::vector<std::string> filenames = {"bigfile1.djctd",
            "bigfile2.djctd", "bigfile3.djctd","bigfile4.djctd",
            "bigfile1.djctd",
                        "bigfile2.djctd", "bigfile3.djctd","bigfile4.djctd",
                        "bigfile1.djctd",
                                    "bigfile2.djctd", "bigfile3.djctd","bigfile4.djctd"};


    trainDataGenerator<float> gen;

    size_t batchsize=100;

    std::cout << "set files" <<std::endl;
    gen.setFileList(filenames);
    gen.setBatchSize(batchsize);

    size_t nepochs=3;
    std::cout << "start" <<std::endl;

    for(size_t e=0;e<nepochs;e++){
        std::cout << "epoch " << e << std::endl;
        //one epoch makes 4 batches
        for(size_t i=0;i<gen.getNBatches();i++){
            auto b = gen.getBatch();
            std::cout << "batch with " << b.nElements() << " elements" <<std::endl;
           // sleep(0.1);
        }
        gen.prepareNextEpoch();
    }

#endif
    return 0;




}
