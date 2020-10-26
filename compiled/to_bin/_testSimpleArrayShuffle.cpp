

#include "../interface/simpleArray.h"
#include <iostream>

int main(){

    using namespace djc;

    std::vector<int64_t> v={2,3,4,52,35,67,3};

    for(auto vv:v)
        std::cout << "v " << vv << std::endl;

    auto rs = simpleArray<float>::splitToDataSplitIndices(v);

    for(auto rss:rs)
        std::cout << "rs " << rss << std::endl;

    auto v2 = simpleArray<float>::dataSplitToSplitIndices(rs);

    for(auto vv:v2)
        std::cout << "v2 " << vv << std::endl;


    //test non ragged shuffle

    simpleArray<float> farr({5,2,1});
    for(float i=0;i<farr.size();i++){
        farr.data()[(int)i]=i;
    }
    std::cout << "input" << std::endl;
    farr.cout();

    auto sfarr = farr.shuffle({4,3,2,1,0});
    sfarr.cout();

    auto ssfarr = sfarr.shuffle({4,3,2,1,0});
    ssfarr.cout();

    bool equal = farr==ssfarr ;
    std::cout << "same? " << equal << std::endl;


    std::cout << "now ragged" <<std::endl;
    std::vector<int64_t> rowsplits = {0,2,5,6,8};

    farr=simpleArray<float> ({4,-1,2},rowsplits);

    std::cout << "size " << farr.size() << std::endl;

    for(float i=0;i<farr.size();i++){
        farr.data()[(int)i]=i;
    }

    farr.cout();

    sfarr = farr.shuffle({3,1,2,0});
    sfarr.cout();

    ssfarr = sfarr.shuffle({3,1,2,0});

    ssfarr.cout();

    equal = ssfarr==farr;
    std::cout << "same? " << equal << std::endl;

    return 0;
}
