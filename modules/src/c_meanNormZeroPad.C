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

int square_bins(
	double xval, double xcenter, 
	int nbins, double half_width) {
	double bin_width = (2*half_width)/nbins;
	double low_edge = xcenter - half_width;
	int ibin = std::floor((xval - low_edge)/bin_width);
	return (ibin >= 0 && ibin < nbins) ? ibin : -1;
}

void particle_binner(
	boost::python::numeric::array numpyarray,
	const boost::python::list inl_norms,
	const boost::python::list inl_means ,
	const boost::python::list inl_branches,
	int nmax, std::string filename, 
	std::string xbranch, std::string xcenter, int xbins, float xwidth, 
	std::string ybranch, std::string ycenter, int ybins, float ywidth
	) {
	std::vector<TString> s_branches = toSTLVector<TString>(inl_branches);
	std::vector<double>  s_norms    = toSTLVector<double>(inl_norms);
	std::vector<double>  s_means    = toSTLVector<double>(inl_means);	

	//get the branch handlers
  //pick all the objects in the collection cut off and zero padding is done per bin
	__hidden::indata branches;
	branches.createFrom(s_branches, s_norms, s_means, MAXBRANCHLENGTH); 

	//get x,y (eta, phi but could be something else) branches
	//mean =0  norm = 1 to avoid scaling
	__hidden::indata xy;
	xy.createFrom({xbranch, ybranch}, {1., 1.}, {0., 0.}, MAXBRANCHLENGTH);

	//get jet center
	__hidden::indata xy_center;
	xy_center.createFrom({xcenter, ycenter}, {1., 1.}, {0., 0.}, 1);

	//get file and tree
	TFile* tfile=new TFile(filename.c_str(), "READ");
	TTree* tree=(TTree*)tfile->Get("deepntuplizer/tree");

	//connect all branches
	branches.setup(tree);
	xy.setup(tree);
	xy_center.setup(tree);

	//find bin indexes
	const int nevents=tree->GetEntries();

	for(int jet=0;jet<nevents;jet++){
		//get values
		branches.zeroAndGet(jet);
		xy.zeroAndGet(jet);
		xy_center.zeroAndGet(jet);

		//map filled indices
		int current_indexes[xbins][ybins];
		for(size_t x=0; x<xbins; x++) {
			for(size_t y=0; y<ybins; y++) {
				current_indexes[x][y] = 0;
			}
		}

		//loop over all candidates
		for(size_t elem=0; elem < MAXBRANCHLENGTH; elem++) {
			//get bin id
			int xidx = square_bins(xy.getData(0, elem), xy_center.getData(0, elem), xbins, xwidth);
			int yidx = square_bins(xy.getData(1, elem), xy_center.getData(1, elem), ybins, ywidth);
			if(xidx == -1 || yidx == -1) continue;
			//if bin is full skip
			if(current_indexes[xidx][yidx] == nmax) continue;
			int particle_idx = current_indexes[xidx][yidx];
			current_indexes[xidx][yidx]++;
			
			for(size_t ifeat=0; ifeat<branches.nfeatures(); ifeat++) {
				numpyarray[jet][xidx][yidx][particle_idx][ifeat]= branches.getData(ifeat, elem);
			}			
		}
		//now, pad with defaults what remains of each bin
		for(size_t x=0; x<xbins; x++) {
			for(size_t y=0; y<ybins; y++) {
				for(size_t idx=current_indexes[x][y]; idx<nmax; idx++) {
					for(size_t ifeat=0; ifeat<branches.nfeatures(); ifeat++) {
						numpyarray[jet][x][y][idx][ifeat] = branches.getDefault(ifeat);
					}
				}
			}
		}

	}	

	tfile->Close();
	delete tfile;
}



// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_meanNormZeroPad) {
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	__hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
	//anyway, it doesn't hurt, just leave this here
	def("process", &process);
	def("particlecluster", &particlecluster);

}
