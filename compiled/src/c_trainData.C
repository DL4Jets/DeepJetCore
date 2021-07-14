/*
 * c_TrainDataInferface.C
 *
 *  Created on: 6 Nov 2019
 *      Author: jkiesele
 */

#include "../interface/trainData.h"


namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace djc;

BOOST_PYTHON_MODULE(c_trainData) {
    Py_Initialize();
    np::initialize();
    using namespace p;
    p::class_<trainData >("trainData")

               .def(self==self)
               .def(self!=self)

        //excplicit overloading
       .def<int (trainData::*)(simpleArray_float32&)>("storeFeatureArray", &trainData::storeFeatureArray)
       .def<int (trainData::*)(simpleArray_int32&)>("storeFeatureArray", &trainData::storeFeatureArray)

       .def<int (trainData::*)(simpleArray_float32&)>("storeTruthArray", &trainData::storeTruthArray)
       .def<int (trainData::*)(simpleArray_int32&)>("storeTruthArray", &trainData::storeTruthArray)

       .def<int (trainData::*)(simpleArray_float32&)>("storeWeightArray", &trainData::storeWeightArray)
       .def<int (trainData::*)(simpleArray_int32&)>("storeWeightArray", &trainData::storeWeightArray)


     //  .def("featureList", &trainData::featureList)
     //  .def("truthList", &trainData::truthList)
     //  .def("weightList", &trainData::weightList)

       .def("nFeatureArrays", &trainData::nFeatureArrays)
       .def("nTruthArrays", &trainData::nTruthArrays)
       .def("nWeightArrays", &trainData::nWeightArrays)

       .def("truncate", &trainData::truncate)
       .def("append", &trainData::append)
       .def("split", &trainData::split)
       .def("nElements", &trainData::nElements)
       .def("readMetaDataFromFile", &trainData::readMetaDataFromFile)

       .def("readFromFile", &trainData::readFromFile)
       .def("readFromFileBuffered", &trainData::readFromFileBuffered)
       .def("writeToFile", &trainData::writeToFile)
       .def("addToFile", &trainData::addToFile)


       .def("copy", &trainData::copy)
       .def("clear", &trainData::clear)
       .def("skim", &trainData::skim)
       .def("getSlice", &trainData::getSlice)

       .def("getNumpyFeatureShapes", &trainData::getNumpyFeatureShapes)
       .def("getNumpyTruthShapes", &trainData::getNumpyTruthShapes)
       .def("getNumpyWeightShapes", &trainData::getNumpyWeightShapes)

       .def("getNumpyFeatureDTypes", &trainData::getNumpyFeatureDTypes)
       .def("getNumpyTruthDTypes", &trainData::getNumpyTruthDTypes)
       .def("getNumpyWeightDTypes", &trainData::getNumpyWeightDTypes)

       .def("getNumpyFeatureArrayNames", &trainData::getNumpyFeatureArrayNames)
       .def("getNumpyTruthArrayNames", &trainData::getNumpyTruthArrayNames)
       .def("getNumpyWeightArrayNames", &trainData::getNumpyWeightArrayNames)

       .def("getTruthRaggedFlags", &trainData::getTruthRaggedFlags)
       .def("transferFeatureListToNumpy", &trainData::transferFeatureListToNumpy)
       .def("transferTruthListToNumpy", &trainData::transferTruthListToNumpy)
       .def("transferWeightListToNumpy", &trainData::transferWeightListToNumpy)


       .def("copyFeatureListToNumpy", &trainData::copyFeatureListToNumpy)
       .def("copyTruthListToNumpy", &trainData::copyTruthListToNumpy)
       .def("copyWeightListToNumpy", &trainData::copyWeightListToNumpy)

;
    ;
}



