/*
 * c_TrainDataInferface.C
 *
 *  Created on: 6 Nov 2019
 *      Author: jkiesele
 */

#define DJC_DATASTRUCTURE_PYTHON_BINDINGS
#include "../interface/trainData.h"


namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace djc;

BOOST_PYTHON_MODULE(c_trainData) {
    Py_Initialize();
    np::initialize();
    p::class_<trainData<float> >("trainData")



       .def("storeFeatureArray", &trainData<float>::storeFeatureArray)
       .def("storeTruthArray", &trainData<float>::storeTruthArray)
       .def("storeWeightArray", &trainData<float>::storeWeightArray)

       .def("featureList", &trainData<float>::featureList)
       .def("truthList", &trainData<float>::truthList)
       .def("weightList", &trainData<float>::weightList)

       .def("nFeatureArrays", &trainData<float>::nFeatureArrays)
       .def("nTruthArrays", &trainData<float>::nTruthArrays)
       .def("nWeightArrays", &trainData<float>::nWeightArrays)

       .def("truncate", &trainData<float>::truncate)
       .def("append", &trainData<float>::append)
       .def("split", &trainData<float>::split)
       .def("nElements", &trainData<float>::nElements)
       .def("readShapesFromFile", &trainData<float>::readShapesFromFile)

       .def("readFromFile", &trainData<float>::readFromFile)
       .def("readFromFileBuffered", &trainData<float>::readFromFileBuffered)
       .def("writeToFile", &trainData<float>::writeToFile)

       .def("clear", &trainData<float>::clear)
       .def("skim", &trainData<float>::skim)

       .def("getKerasFeatureShapes", &trainData<float>::getKerasFeatureShapes)
       .def("getKerasFeatureDTypes", &trainData<float>::getKerasFeatureDTypes)

       .def("getTruthRaggedFlags", &trainData<float>::getTruthRaggedFlags)
       .def("transferFeatureListToNumpy", &trainData<float>::transferFeatureListToNumpy)
       .def("transferTruthListToNumpy", &trainData<float>::transferTruthListToNumpy)
       .def("transferWeightListToNumpy", &trainData<float>::transferWeightListToNumpy)


       .def("copyFeatureListToNumpy", &trainData<float>::copyFeatureListToNumpy)
       .def("copyTruthListToNumpy", &trainData<float>::copyTruthListToNumpy)
       .def("copyWeightListToNumpy", &trainData<float>::copyWeightListToNumpy)

;
    ;
}



