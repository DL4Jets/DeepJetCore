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
#include <vector>
#include "TFile.h"
#include "TTree.h"
//don't use new root6 stuff, has problems with leaf_list
//#include "TTreeReader.h"
//#include "TTreeReaderValue.h"
//#include "TTreeReaderArray.h"
//#include "TTreeReaderUtils.h"
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "TStopwatch.h"
#include "../interface/indata.h"
#include "../interface/pythonToSTL.h"
#include "../interface/helper.h"

using namespace boost::python; //for some reason....

enum modeen {en_flat,en_particlewise};
//can be extended later to run on a list of deep AND flat branches. Start simple first

// Functions to demonstrate extraction

void priv_meanNormZeroPad(boost::python::numeric::array& numpyarray,
		std::vector<__hidden::indata>   data,
		TFile* tfile, modeen mode);


void priv_process(boost::python::numeric::array numpyarray,
		const boost::python::list inl_norms,
		const boost::python::list inl_means ,
		const boost::python::list inl_branches,
		const boost::python::list lmaxs,
		std::string filename, enum modeen mode);





/*
 * wrapper to create input to C++ only function
 * Can be generalised to doing it at the same time for many different sized branches
 */
void process(boost::python::numeric::array numpyarray,
		const boost::python::list inl_norms,
		const boost::python::list inl_means ,
		const boost::python::list inl_branches,
		const boost::python::list lmaxs,
		std::string filename) {



	//Py_Initialize();
	priv_process(numpyarray,inl_norms,inl_means,inl_branches,lmaxs,filename,en_flat);

}

void particlecluster(boost::python::numeric::array numpyarray,
		const boost::python::list inl_norms,
		const boost::python::list inl_means ,
		const boost::python::list inl_branches,
		const boost::python::list lmaxs,
		std::string filename) {


	if(len(inl_branches)>1)
		throw std::runtime_error("particlecluster only possible for one collection of same type at a time");
	//Py_Initialize();
	priv_process(numpyarray,inl_norms,inl_means,inl_branches,lmaxs,filename,en_particlewise);

}


void priv_process(boost::python::numeric::array numpyarray,
		const boost::python::list inl_norms,
		const boost::python::list inl_means ,
		const boost::python::list inl_branches,
		const boost::python::list lmaxs,
		std::string filename, enum modeen mode){


	/*
	 * ******* convert the python lists to stl containers
	 */

	std::vector< std::vector<TString>  >s_branches_ = toSTL2DVector<TString>(inl_branches);
	std::vector< std::vector<double> > s_norms = toSTL2DVector<double>(inl_norms);
	std::vector< std::vector<double> > s_means = toSTL2DVector<double>(inl_means);
	std::vector<int> s_max = toSTLVector<int>(lmaxs);



	std::vector<__hidden::indata> alldata;
	alldata=__hidden::createDataVector(s_branches_,s_norms,s_means,s_max);

	TString tfilename=filename;
	//this is a bit more stable and possibly faster
	//root version seems to not support xrootd
	//tfilename=prependXRootD(tfilename);

	TFile * tfile=new TFile(tfilename,"READ");

	priv_meanNormZeroPad(numpyarray,alldata,tfile,mode);

	tfile->Close();
	delete tfile;

}

//root-only functions
//change all inputs except for in_data to vectors for simultaneous use
void priv_meanNormZeroPad(boost::python::numeric::array& numpyarray,
		std::vector<__hidden::indata>   datacollection,
		TFile* tfile, modeen mode){

	//for checks
	int ysize=0;
	for(const auto& d:datacollection)
		ysize+= d.max*d.branches.size();


	TStopwatch stopw;

	//all branches are floats!
	TTree* tree=(TTree*)tfile->Get("deepntuplizer/tree");

	for(auto& d:datacollection)
		d.setup(tree);

	//std::cout << "looping over events: "<< stopw.RealTime () <<std::endl;
	//stopw.Reset();
	//stopw.Start();
	const int nevents=tree->GetEntries();
	const int datasize=datacollection.size();

	for(int jet=0;jet<nevents;jet++){
		for(auto& d:datacollection)
			d.allZero();
		for(auto& d:datacollection)
			d.getEntry(jet);


		for(size_t c=0;c<datasize;c++){
			const size_t& doffset=datacollection.at(c).offset_;

			for(int b=0;b<datacollection.at(c).branches.size();b++){
				const size_t boffset=datacollection.at(c).branchOffset(b);
				for(int i=0;i<datacollection.at(c).getMax();i++){
					if(mode==en_flat){
						size_t listindex=i+doffset+boffset;
						numpyarray[jet][listindex]= datacollection.at(c).getData(b,i);
					}
					else if(mode==en_particlewise){
						//c is 0, only
						numpyarray[jet][i][b]= datacollection.at(c).getData(b,i);

					}
				}
			}
		}
	}
}


void priv_particlecluster(boost::python::numeric::array& numpyarray,
		std::vector<__hidden::indata>   datacollection,
		TFile* tfile){

}


// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_meanNormZeroPad) {
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	__hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
	//anyway, it doesn't hurt, just leave this here
	def("process", &process);
	def("particlecluster", &particlecluster);

}
