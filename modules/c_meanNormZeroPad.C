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

#define MAXBRANCHLENGTH 200

using namespace boost::python; //for some reason....

namespace __hidden{
class indata{
public:
	indata():max(0),offset_(0){

	}
	void setSize(size_t i){
		norms.resize(i,1);
		means.resize(i,0);
		branches.resize(i);
		tbranches.resize(i,0);
		buffer.resize(i,0);
	}

	~indata(){
		//	for(auto& b:buffer)
		//		if(b) delete b;
		//	for(auto& b:branches)
		//		if(b) delete b;
	}

	size_t getMax()const{
		return max;
	}

	float getData(const size_t& b,const size_t& i){
		float ret= buffer.at(b)[i];
		ret -= means.at(b);
		ret /= norms.at(b);
		return ret;
	}

	void allZero(){
		for(auto& c:buffer)
			for(int i=0;i<max;i++)
				c[i]=0;
	}

	void getEntry(size_t entry){
		for(auto& b:tbranches)
			b->GetEntry(entry);
	}

	void setup(TTree* tree){
		if(MAXBRANCHLENGTH<max)
			throw std::runtime_error("max larger than buffer! (clean up here needed: TBI)");

		for(auto& b: buffer)
			b=new float[MAXBRANCHLENGTH];

		for(size_t i=0;i<branches.size();i++){
			tbranches.at(i)=new TBranch();
			tree->SetBranchAddress(branches.at(i),buffer.at(i),&tbranches.at(i));
		}
	}

	size_t branchOffset(const size_t& i){
		return i*max;
	}

	std::vector<float* > buffer;

	std::vector<TBranch* > tbranches;

	size_t offset_;

	std::vector<float> norms,means;
	std::vector<TString> branches;
	int max;
};
}

//can be extended later to run on a list of deep AND flat branches. Start simple first

// Functions to demonstrate extraction

void priv_meanNormZeroPad(boost::python::numeric::array& numpyarray,
		std::vector<__hidden::indata>   data,
		TFile* tfile);

std::vector<__hidden::indata> createData(std::vector< std::vector<TString>  >s_branches,
		std::vector< std::vector<double> > s_norms,
		std::vector< std::vector<double> > s_means,
		std::vector<int> s_max
){

	std::vector<__hidden::indata>  alldata;
	size_t offset=0;
	for(int i=0;i<s_branches.size();i++){
		size_t branchlength=s_branches.at(i).size();
		__hidden::indata data_config;
		data_config.setSize(branchlength);

		data_config.max = s_max.at(i);
		for(int j=0;j<s_branches.at(i).size();j++){
			data_config.branches.at(j)=s_branches.at(i).at(j);
			data_config.norms.at(j)=s_norms.at(i).at(j);
			data_config.means.at(j)=s_means.at(i).at(j);
		}
		data_config.offset_=offset;
		alldata.push_back(data_config);

		offset+=data_config.branches.size()*data_config.max;
	}


	return alldata;
}

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

	// copy to be safe... somehow weird things happened here...
	// this is also the reason for copying first to vectors before doing anything more
	//
	boost::python::list in_norms(inl_norms);
	boost::python::list in_means(inl_means);
	boost::python::list in_branches(inl_branches);
	boost::python::list maxs(lmaxs);

	if(boost::python::len(in_branches)!=boost::python::len(in_norms) ||
			boost::python::len(in_norms)!=boost::python::len(in_means) ||
			boost::python::len(maxs) != boost::python::len(in_branches)){
		std::cout << "lists must have same size" <<std::endl;
		throw std::runtime_error("meanNormZeroPad: lists must have same size");
	}


	/*
	 * ******* convert the python lists to stl containers
	 */

	std::vector< std::vector<TString>  >s_branches_;
	std::vector< std::vector<double> > s_norms,s_means;
	std::vector<int> s_max;
	//the 2d lists seem very picky. don't do anything fancy and just convert
	size_t fulllength=boost::python::len(in_branches);
	for(int i=0;i<fulllength;i++){
		if(boost::python::len(in_branches[i])!=boost::python::len(in_norms[i]) ||
				boost::python::len(in_norms[i])!=boost::python::len(in_means[i])){
			std::cout << "lists must have same size" <<std::endl;
			throw std::runtime_error("meanNormZeroPad: lists must have same size");
		}
		std::vector<TString> tmps;
		std::vector<double> tmpn,tmpm;
		size_t seclength=boost::python::len(in_branches[i]);
		int max=boost::python::extract<int>(maxs[i]);
		s_max.push_back(max);
		for(int j=0;j<seclength;j++){
			std::string s=boost::python::extract<std::string>(in_branches[i][j]);
			tmps.push_back(s);
			double norm=boost::python::extract<double>(in_norms[i][j]);
			double mean=boost::python::extract<double>(in_means[i][j]);
			tmpn.push_back(norm);
			tmpm.push_back(mean);
		}
		s_norms.push_back(tmpn);
		s_means.push_back(tmpm);
		s_branches_.push_back(tmps);
	}


	/*
	 * ******* prepare and invoke function doing the actual work
	 */

	std::vector<__hidden::indata> alldata;
	alldata=createData(s_branches_,s_norms,s_means,s_max);


	//check if file system is ok
	//a bit of protection agains eos glitches
	/* still some lib issues, to be resolved later FIXME
	boost::filesystem::path dir=filename.data();//if we use boost anyway
	dir=dir.parent_path();
	size_t seccounter=0;
	while (! boost::filesystem::exists(dir)){
		sleep(1);
		seccounter++;
		if(seccounter>119)//give two minutes to recover - shoul dbe more than enough
			throw std::runtime_error("c_meanNormZeroPad:process: file system not available");
	}
	 */
	// ..and hope no glitches occour when reading....
	TFile * tfile=new TFile(filename.data(),"READ");

	priv_meanNormZeroPad(numpyarray,alldata,tfile);

	tfile->Close();
	delete tfile;

}

//root-only functions
//change all inputs except for in_data to vectors for simultaneous use
void priv_meanNormZeroPad(boost::python::numeric::array& numpyarray,
		std::vector<__hidden::indata>   datacollection,
		TFile* tfile){

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

					size_t listindex=i+doffset+boffset;
					//direct memory access to numpy array
					//be careful that it is really float32 (4 Byte)!! true on unix systems
					numpyarray[jet][listindex]= datacollection.at(c).getData(b,i);
				}
			}
		}
	}
}





// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_meanNormZeroPad) {
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	__hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
	//anyway, it doesn't hurt, just leave this here
	def("process", &process);

}
