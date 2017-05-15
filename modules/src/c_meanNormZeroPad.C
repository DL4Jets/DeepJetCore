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
    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
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
        int nmax, std::string filename, std::string counter_branch,
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

    //get particle counter
    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);

    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get("deepntuplizer/tree");

    //connect all branches
    branches.setup(tree);
    xy.setup(tree);
    xy_center.setup(tree);
    counter.setup(tree);

    // std::cout << "looping over events for " << counter_branch <<std::endl;
    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
    TStopwatch stopw;
    for(int jet=0;jet<nevents;jet++){
        // if(jet % 100 == 0) {
        // 	std::cout << "filling binned jet: " << jet << " (" << stopw.RealTime()/100. << " sec./evt)" << std::endl;
        // 	stopw.Reset();
        // 	stopw.Start();
        // }
        //get values
        branches.zeroAndGet(jet);
        xy.zeroAndGet(jet);
        xy_center.zeroAndGet(jet);
        counter.zeroAndGet(jet);

        //std::cout << "jet #" << jet << ": " << xy_center.getData(0, 0) << ", " << xy_center.getData(1, 0) << std::endl;

        //map filled indices
        int current_indexes[xbins][ybins];
        //now, pad with defaults every bin (for safety up here)
        for(size_t x=0; x<xbins; x++) {
            for(size_t y=0; y<ybins; y++) {
                current_indexes[x][y] = 0;
                for(size_t idx=0; idx<nmax; idx++) {
                    for(size_t ifeat=0; ifeat<branches.nfeatures(); ifeat++) {
                        numpyarray[jet][x][y][idx][ifeat] = branches.getDefault(ifeat);
                    }
                }
            }
        }

        //loop over all candidates
        int ncharged = counter.getData(0, 0);
        for(size_t elem=0; elem < ncharged; elem++) {
            //get bin id
            int xidx = square_bins(xy.getData(0, elem), xy_center.getData(0, 0), xbins, xwidth);
            int yidx = square_bins(xy.getData(1, elem), xy_center.getData(1, 0), ybins, ywidth);
            if(xidx == -1 || yidx == -1) continue;
            //if bin is full skip
            if(current_indexes[xidx][yidx] == nmax) continue;
            int particle_idx = current_indexes[xidx][yidx];
            current_indexes[xidx][yidx]++;

            for(size_t ifeat=0; ifeat<branches.nfeatures(); ifeat++) {
                double feature_value = branches.getData(ifeat, elem);
                // if(ifeat == 0)
                // 	std::cout << "Jet: " << jet << " Candidate: " << elem << ", bin (" << xidx << ", " << yidx << ", " << particle_idx
                // 						<< ") feat #"<< ifeat <<": " << feature_value << std::endl;
                numpyarray[jet][xidx][yidx][particle_idx][ifeat]= feature_value;
            }
        }

        //BUG HUNTING
        //int filled = 0;
        //for(size_t x=0; x<xbins; x++) {
        //	for(size_t y=0; y<ybins; y++) {
        //		filled += current_indexes[x][y];
        //	}
        //}
        //std::cout << "jet #" << jet << " had " << ncharged << " candidates. I filled " << filled << std::endl;
    }

    tfile->Close();
    delete tfile;
}

void fillDensityMap(boost::python::numeric::array numpyarray,
        double norm,
        std::string in_branch,
        std::string in_weightbranch,
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth,
        double offset
        ){


    TString branchstr=in_branch;
    TString weightstr=in_weightbranch;

    __hidden::indata branch;
    branch.createFrom({branchstr}, {norm}, {0.}, MAXBRANCHLENGTH);
    //only normalisation, no mean substr.

    bool useweights=false;
    if(weightstr.Length())
        useweights=true;

    __hidden::indata weight;
    if(useweights)
        weight.createFrom({weightstr}, {1.}, {0.}, MAXBRANCHLENGTH);

    //get x,y (eta, phi but could be something else) branches
    //mean =0  norm = 1 to avoid scaling
    __hidden::indata xy;
    xy.createFrom({xbranch, ybranch}, {1., 1.}, {0., 0.}, MAXBRANCHLENGTH);

    //get jet center
    __hidden::indata xy_center;
    xy_center.createFrom({xcenter, ycenter}, {1., 1.}, {0., 0.}, 1);

    //get particle counter
    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);

    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get("deepntuplizer/tree");

    //connect all branches
    branch.setup(tree);
    if(useweights)
        weight.setup(tree);
    xy.setup(tree);
    xy_center.setup(tree);
    counter.setup(tree);

    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
    for(int jet=0;jet<nevents;jet++){

        branch.zeroAndGet(jet);
        if(useweights)
            weight.zeroAndGet(jet);
        xy.zeroAndGet(jet);
        xy_center.zeroAndGet(jet);
        counter.zeroAndGet(jet);

        std::vector<std::vector<float> > densemap(xbins,std::vector<float>(ybins,0));

        int ncharged = counter.getData(0, 0);
        for(size_t elem=0; elem < ncharged; elem++) {
            //get bin id
            int xidx = square_bins(xy.getData(0, elem), xy_center.getData(0, 0), xbins, xwidth);
            int yidx = square_bins(xy.getData(1, elem), xy_center.getData(1, 0), ybins, ywidth);
            if(xidx == -1 || yidx == -1) continue;

            float feature_value = branch.getData(0, elem)-offset;
            //std::cout << elem << xidx << "  " << yidx << " "<< feature_value <<std::endl;
            float weight_value =1;
            if(useweights)
                weight_value = weight.getData(0, elem);

            densemap.at(xidx).at(yidx)+=feature_value*weight_value;

        }

        //seems to need burte-force way because of boost:numpy
        for(int i=0;i<xbins;i++){
            for(int j=0;j<ybins;j++){
                numpyarray[jet][i][j]=densemap.at(i).at(j);
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
    def("particle_binner", &particle_binner);
    def("fillDensityMap", &fillDensityMap);
}
