#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
//#include "boost/filesystem.hpp"
#include <iostream>
#include <stdint.h>
#include "TString.h"
#include <string>
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "../interface/pythonToSTL.h"
#include "friendTreeInjector.h"
#include "rocCurveCollection.h"

using namespace boost::python; //for some reason....



void makeROCs(
		std::string intextfile,
		const boost::python::list names,
		const boost::python::list probabilities ,
		const boost::python::list truths,
		const boost::python::list vetos,
		const boost::python::list colors,
		std::string outfile,
		std::string cuts) {


	std::vector<TString>  s_names = toSTLVector<TString>(names);
	std::vector<TString>  s_probabilities = toSTLVector<TString>(probabilities);
	std::vector<TString>  s_truths = toSTLVector<TString>(truths);
	std::vector<TString>  s_vetos = toSTLVector<TString>(vetos);
	std::vector<TString>  s_colors = toSTLVector<TString>(colors);


	/*
	 * Size checks!!!
	 */
	if(s_names.size() != s_probabilities.size() ||
			s_names.size() != s_truths.size()||
			s_names.size() != s_vetos.size()||
			s_names.size() != s_colors.size())
		throw std::runtime_error("makeROCs: input lists must have same size");

	friendTreeInjector injector;
	injector.addFromFile((TString)intextfile);
	injector.createChain();

	rocCurveCollection rocs;

	for(size_t i=0;i<s_names.size();i++){
		rocs.addROC(s_names.at(i),s_probabilities.at(i),s_truths.at(i),s_vetos.at(i),s_colors.at(i),(TString)cuts);
	}

	rocs.printRocs(injector.getChain(),(TString)outfile);

}




// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_makeROCs) {
	//__hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
	//anyway, it doesn't hurt, just leave this here
	def("makeROCs", &makeROCs);

}
