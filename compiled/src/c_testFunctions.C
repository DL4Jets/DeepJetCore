
#include <boost/python.hpp>
#include <boost/python/exception_translator.hpp>

//tests
#include "trainDataFileStreamer.h"


using namespace boost::python;

BOOST_PYTHON_MODULE(c_testFunctions) {
    def("testTrainDataFileStreamer", &djc::test::testTrainDataFileStreamer);

}
