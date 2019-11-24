

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


    std::vector<int64_t> rowsplits = {0,2,5,6,8};

    simpleArray<float> farr({4,-1,2},rowsplits);

    std::cout << "size " << farr.size() << std::endl;

    for(float i=0;i<farr.size();i++){
        farr.data()[(int)i]=i;
    }

    farr.cout();

    farr.writeToFile("testfile.djcd");



    simpleArray<float> farr2;
    farr2.readFromFile("testfile.djcd");

    farr2.cout();

    return 1;

    for(float i=0;i<farr2.size();i++){
            farr2.data()[(int)i]=i;
            std::cout << i << std::endl;
        }
    std::cout << "copied row splits" << std::endl;
    auto farr2rs = farr2.rowsplits();
    for(auto rs: farr2rs)
        std::cout << rs << std::endl;

    std::cout << "access" << std::endl;

    std::cout << "0 , 0, 0 " << farr2.at(0,0,0) << std::endl;
    std::cout << "0 , 1, 0 " << farr2.at(0,1,0) << std::endl;

    std::cout << "3 , 1, 0 " << farr2.at(3,1,0) << std::endl;
    std::cout << "3 , 0, 0 " << farr2.at(3,0,0) << std::endl;

    farr2.at(3,1,0) = 13;
    std::cout << "3 , 1, 0 " << farr2.at(3,1,0) << std::endl;
    std::cout << "3 , 1, 1 " << farr2.at(3,1,1) << std::endl;


    auto farr3 = farr2.split(2);

    for(auto s : farr2.shape())
        std::cout << s <<" " ;
    std::cout <<  std::endl;

    for(auto s : farr3.shape())
        std::cout << s <<" " ;
    std::cout <<  std::endl;

    std::cout << farr3.at(0,1,0) << " = 2  "  << std::endl;
    std::cout << farr2.at(1,0,0) << " = 12  "  << std::endl;


    std::cout << "append again" <<std::endl;

    coutExpected();

   // farr3.append(farr2);
    auto farr4 = farr3;
    farr3.append(farr3);
    for(auto s : farr3.shape())
        std::cout << s <<" " ;
    std::cout <<  std::endl;
    for(size_t i=0;i<farr3.size();i++){
        std::cout << farr3.data()[i] << " ";
    }
    std::cout <<  std::endl;


    std::cout << "double split not ragged" <<std::endl;

    simpleArray<float> nors({20,20,3});
    nors = nors.split(10);
    coutarray(nors);
    nors = nors.split(5);
    coutarray(nors);
    nors = nors.split(1);
    coutarray(nors);
    std::cout << "double split  ragged" <<std::endl;

    auto rsdsp = farr;
    coutarray(rsdsp);
    rsdsp = rsdsp.split(2);
    coutarray(rsdsp);
    rsdsp = rsdsp.split(1);
    coutarray(rsdsp);


}
