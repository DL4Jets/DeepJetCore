

#include <iostream>
#include "../interface/quicklzWrapper.h"
#include "../interface/simpleArray.h"
#include "../interface/trainData.h"

#include "../interface/trainDataGenerator.h"

#include <iostream>
using namespace djc;

void coutExpected(){
    std::cout << "expected shape:\n[ \n[[0,1], [2,3]],\n[[3,4], [5,6], [7,8]],\n[[9,10]],\n[[11,12],[13,14]]\n]" << std::endl;

}

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

#ifdef IFONREFORNOW
    std::vector<size_t> rowsplits = {0,2,5,6,8};

    trainData<float> td;

    size_t idx = td.storeFeatureArray(simpleArray<float>({4,-1,2},rowsplits));

    simpleArray<float> & farr = td.featureArray(idx);

    std::cout << "size " << farr.size() << std::endl;

    for(float i=0;i<farr.size();i++){
        farr.data()[(int)i]=i;
        std::cout << i << std::endl;
    }
    /*
    std::cout << std::endl;
    std::cout << "append to self" << std::endl;
    td.append(td);
    std::cout << "appended to self" << std::endl;
    coutarray(farr);
*/

    //first try without row splits
    trainData<float> td_nrs;
    td_nrs.storeFeatureArray(simpleArray<float>({100,20,20}));
    td_nrs = td_nrs.split(20);
    coutarray(td.featureArray(idx));
    td_nrs = td_nrs.split(10);
    coutarray(td.featureArray(idx));
    td_nrs = td_nrs.split(3);
    coutarray(td.featureArray(idx));


    td = td.split(2);

    std::cout << "split  self, size after " << td.nElements() << std::endl;
    coutarray(td.featureArray(idx));
    std::cout << "split self again at 1" << std::endl;

    //this segfaults....
    td = td.split(1);
    std::cout << "split  self, size after " << td.nElements() << std::endl;
    coutarray(td.featureArray(idx));

#endif

}
