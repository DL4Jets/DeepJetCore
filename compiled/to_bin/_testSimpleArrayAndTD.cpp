

#include <iostream>
#include "../interface/quicklzWrapper.h"
#include "../interface/simpleArray.h"
#include "../interface/trainData.h"

#include "../interface/trainDataGenerator.h"

using namespace djc;

void coutarray(const simpleArray<float> & farr){
    std::cout << "data size "<< farr.size() <<std::endl;
    for(int i=0;i<farr.size();i++){
        std::cout << farr.data()[i] << ", ";
    }
    std::cout << std::endl;
    for(const auto s: farr.shape())
        std::cout << s << ", ";
    std::cout << "\nrow splits " << std::endl;
    if(farr.rowsplits().size()){
        for(const auto s: farr.rowsplits())
            std::cout << s << ", ";
    }
    std::cout << std::endl;
}


int main(){

    simpleArray<float> farr({5,2,1});
    for(float i=0;i<farr.size();i++){
        farr.data()[(int)i]=i;
     std::cout << i << std::endl;
    }
    std::cout << std::endl;


    //write part
    FILE *ofile = fopen("testfile.djcd", "wb");

    farr.addToFile(ofile);

    fclose(ofile);
    //read part
    FILE *ifile = fopen("testfile.djcd", "rb");

    simpleArray<float> ifarr;

    ifarr.readFromFile(ifile);


    fclose(ifile);


    for(size_t i=0;i<ifarr.shape().size();i++)
        std::cout << ifarr.shape()[i] << std::endl;
    std::cout << std::endl;


    for(size_t i=0;i<ifarr.shape().size();i++)
        std::cout << ifarr.shape()[i] << std::endl;

    auto arrs = ifarr.split(3);
    std::cout << "splitting stuff " <<std::endl;

    std::cout << std::endl;
    for(size_t i=0;i<arrs.shape().size();i++)
        std::cout << arrs.shape()[i] << std::endl;
    std::cout << std::endl;

    for(size_t i=0;i<ifarr.size();i++)
        std::cout << ifarr.data()[i] << std::endl;
    std::cout << std::endl;
    for(size_t i=0;i<arrs.size();i++)
        std::cout << arrs.data()[i] << std::endl;

    ifarr.append(arrs);

    std::cout << "appended "<< std::endl;
    for(size_t i=0;i<ifarr.shape().size();i++)
        std::cout << ifarr.shape()[i] << std::endl;

    std::cout << std::endl;
    for(size_t i=0;i<ifarr.size();i++)
        std::cout << ifarr.data()[i] << std::endl;

    std::cout << "copy and move "<< std::endl;

    auto ar2 = ifarr;
    for(size_t i=0;i<ar2.shape().size();i++)
        std::cout << ar2.shape()[i] << std::endl;
    std::cout << std::endl;
    for(size_t i=0;i<ar2.size();i++)
        std::cout << ar2.data()[i] << std::endl;

    std::cout << "vector "<< std::endl;
    std::vector<simpleArray<float> > vec;
    vec.push_back(ar2);

    auto & var2 = vec.at(0);

    for(size_t i=0;i<var2.shape().size();i++)
        std::cout << var2.shape()[i] << std::endl;
    std::cout << std::endl;
    for(size_t i=0;i<var2.size();i++)
        std::cout << var2.data()[i] << std::endl;

    return 1;

    std::cout << "trainData Stuff "<< std::endl;

    trainData<float> td;

    size_t fvidx = td.addFeatureArray( {30,2,3} );
    std::cout << fvidx << std::endl;
    std::cout << td.featureArray(fvidx).size() << std::endl;
    td.featureArray(fvidx).at(0,0,0) = 1;
    td.featureArray(fvidx).at(0,0,1) = 1;
    td.featureArray(fvidx).at(0,1,0) = 1;
    td.featureArray(fvidx).at(1,0,0) = 1;
    td.featureArray(fvidx).at(1,1,0) = 1;
    td.featureArray(fvidx).at(2,1,2) = 2;

    std::cout << "print "<< std::endl;

    for(size_t i=0;i<td.featureArray(fvidx).size();i++)
            std::cout << td.featureArray(fvidx).data()[i] << std::endl;


    size_t tidx = td.addTruthArray( {30,1} );
    std::cout << "written "<< std::endl;

    for(size_t i=0;i<td.truthArray(tidx).size();i++)
            std::cout << td.truthArray(tidx).data()[i] << std::endl;

    td.truncate(10);

    td.writeToFile("tempfile.djctd");

    trainData<float> td2;
    td2.readFromFile("tempfile.djctd");

    std::cout << "read "<< std::endl;

    for(size_t i=0;i<td2.truthArray(tidx).size();i++)
            std::cout << td2.truthArray(tidx).data()[i] << std::endl;

    td2.append(td);

    std::cout << "appended "<< std::endl;

    for(size_t i=0;i<td2.truthArray(tidx).size();i++)
            std::cout << td2.truthArray(tidx).data()[i] << std::endl;


    //big data test

    trainData<float> bigtd;

    size_t fidx = bigtd.addFeatureArray({1000, 4000, 10});
    for(size_t i=0;i<bigtd.featureArray(fidx).size();i++)
        bigtd.featureArray(fidx).data()[i]=i;

    bigtd.writeToFile("bigfile.djctd");
    bigtd.clear();

    std::cout << "reading file "<< std::endl;

    trainData<float> rbig;
    rbig.readFromFile("bigfile.djctd");
    std::cout << "done reading file "<< std::endl;

}
