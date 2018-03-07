//allows functions with 18 or less paramenters
#define BOOST_PYTHON_MAX_ARITY 20
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
#include <cmath>

using namespace boost::python; //for some reason....

static TString treename="deepntuplizer/tree";


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
    TTree* tree=(TTree*)tfile->Get(treename);

    for(auto& d:datacollection)
        d.setup(tree,treename);

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
        int nbins, double half_width,
        bool isPhi=false) {
    double bin_width = (2*half_width)/nbins;
    double low_edge = 0;
    if(isPhi)
        low_edge =deltaPhi( xcenter, half_width);
    else
        low_edge = xcenter - half_width;
    int ibin = 0;
    if(isPhi)
        ibin=std::floor((double)deltaPhi(xval,low_edge)/bin_width);
    else
        ibin=std::floor((xval - low_edge)/bin_width);
    return (ibin >= 0 && ibin < nbins) ? ibin : -1;
}
bool branchIsPhi(std::string branchname){
    TString bn=branchname;
    bn.ToLower();
    return bn.Contains("phi");
}

void particle_binner(
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth,
        //binned variables
        boost::python::numeric::array numpyarray,
        const boost::python::list inl_norms,
        const boost::python::list inl_means ,
        const boost::python::list inl_branches,
        int nmax,
        //summed variables
        boost::python::numeric::array sum_npy_array,
        const boost::python::list sum_inl_norms,
        const boost::python::list sum_inl_means ,
        const boost::python::list summed_branches
) {
    std::vector<TString> s_branches = toSTLVector<TString>(inl_branches);
    std::vector<double>  s_norms    = toSTLVector<double>(inl_norms);
    std::vector<double>  s_means    = toSTLVector<double>(inl_means);
    std::vector<TString> s_sum_branches = toSTLVector<TString>(summed_branches);
    std::vector<double>  s_sum_stds     = toSTLVector<double>(sum_inl_norms);
    std::vector<double>  s_sum_means    = toSTLVector<double>(sum_inl_means);

    //get the branch handlers
    //pick all the objects in the collection cut off and zero padding is done per bin
    __hidden::indata branches;
    branches.createFrom(s_branches, s_norms, s_means, MAXBRANCHLENGTH);

    //pick branches to be summed by bin
    std::vector<double> dummy(s_sum_branches.size(), 0.);
    __hidden::indata sum_branches;
    sum_branches.createFrom(s_sum_branches, dummy, dummy, MAXBRANCHLENGTH);

    //get x,y (eta, phi but could be something else) branches
    //mean =0  norm = 1 to avoid scaling
    __hidden::indata xy;
    xy.createFrom({xbranch, ybranch}, {1., 1.}, {0., 0.}, MAXBRANCHLENGTH);
    bool XisPhi=branchIsPhi(xbranch);
    bool YisPhi=branchIsPhi(ybranch);

    //get jet center
    __hidden::indata xy_center;
    xy_center.createFrom({xcenter, ycenter}, {1., 1.}, {0., 0.}, 1);

    //get particle counter
    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);

    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get(treename);

    //connect all branches
    branches.setup(tree);
    sum_branches.setup(tree);
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
        sum_branches.zeroAndGet(jet);
        xy.zeroAndGet(jet);
        xy_center.zeroAndGet(jet);
        counter.zeroAndGet(jet);

        //map summed features, for some reasons getting a value out of a numpy array is tough (or so I was told)
        double summed_values[xbins][ybins][sum_branches.nfeatures()+1];
        //map filled indices
        int current_indexes[xbins][ybins];
        //now, pad with defaults every bin (for safety up here)
        for(size_t x=0; x<xbins; x++) {
            for(size_t y=0; y<ybins; y++) {
                current_indexes[x][y] = 0;

                for(size_t ifeat=0; ifeat < (sum_branches.nfeatures()+1); ifeat++) {
                    summed_values[x][y][ifeat] = 0;
                }

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
            int xidx = square_bins(xy.getData(0, elem), xy_center.getData(0, 0), xbins, xwidth,XisPhi);
            int yidx = square_bins(xy.getData(1, elem), xy_center.getData(1, 0), ybins, ywidth,YisPhi);
            if(xidx == -1 || yidx == -1) continue;

            //bin summing
            summed_values[xidx][yidx][0]++;
            for(size_t ifeat=1; ifeat < (sum_branches.nfeatures()+1); ifeat++) {
                double feat_val = sum_branches.getRaw(ifeat-1, elem);
                summed_values[xidx][yidx][ifeat] += feat_val;
            }

            //single values
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

        for(size_t x=0; x<xbins; x++) {
            for(size_t y=0; y<ybins; y++) {
                for(size_t ifeat=0; ifeat < (sum_branches.nfeatures()+1); ifeat++) {
                    //hardcoded scaling! to change if zero padding method changes!
                    sum_npy_array[jet][x][y][ifeat] = (summed_values[x][y][ifeat] - s_sum_means.at(ifeat)) / s_sum_stds.at(ifeat);
                }
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
void priv_fillDensityMap(boost::python::numeric::array numpyarray,
        double norm,
        std::string in_branch,
        std::string in_weightbranch,
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth,
        double offset, bool count=false
);

void fillDensityMap(boost::python::numeric::array numpyarray,
        double norm,
        std::string in_branch,
        std::string in_weightbranch,
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth,
        double offset
){
    priv_fillDensityMap(numpyarray,norm,in_branch,in_weightbranch,filename,counter_branch,
            xbranch,xcenter,xbins,xwidth,
            ybranch,ycenter,ybins,ywidth,offset,false);
}
void fillCountMap(boost::python::numeric::array numpyarray,
        double norm,
        std::string in_weightbranch,
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth,
        double offset
){
    std::string in_branch="";
    priv_fillDensityMap(numpyarray,norm,in_branch,in_weightbranch,filename,counter_branch,
            xbranch,xcenter,xbins,xwidth,
            ybranch,ycenter,ybins,ywidth,offset,true);
}

void fillDensityLayers(boost::python::numeric::array numpyarray,
        const boost::python::list  inl_norms,
        const boost::python::list  inl_means,
        const boost::python::list  in_branches,
        const boost::python::list  modes,
        std::string layer_branch,
        int maxlayers,
        int layer_offset,
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth
){

    std::vector<TString> s_branches = toSTLVector<TString>(in_branches);
    std::vector<TString> s_modes = toSTLVector<TString>(modes);
    std::vector<double>  s_norms    = toSTLVector<double>(inl_norms);
    std::vector<double>  s_means   = toSTLVector<double>(inl_means);

    enum en_fillmodes{fm_sum,fm_average,fm_single, fm_relXsingle,fm_relYsingle};
    std::vector<en_fillmodes> fillmodes(s_modes.size());
    for(size_t i=0;i<fillmodes.size();i++){
        const TString& mode=s_modes.at(i);
        if(mode=="sum")
            fillmodes.at(i)=fm_sum;
        else if(mode=="average")
            fillmodes.at(i)=fm_average;
        else if(mode=="single")
            fillmodes.at(i)=fm_single;
        else if(mode=="relXsingle")
            fillmodes.at(i)=fm_relXsingle;
        else if(mode=="relYsingle")
            fillmodes.at(i)=fm_relYsingle;
        else
            throw std::runtime_error("fillDensityLayers: fill mode not recognised");
    }


    //in case the layer branch has also been used as a feature branch, it will be overwritten later
    //define mask to fall back to layer branch read
    std::vector<TString>::iterator it=std::find(s_branches.begin(),s_branches.end(),(TString)layer_branch);
    int maskedlayerbranch=-1;
    if(it!=s_branches.end()){
        maskedlayerbranch=it-s_branches.begin();
    }



    __hidden::indata branch;
    branch.setMask(maskedlayerbranch);
    branch.createFrom(s_branches, s_norms, s_means, MAXBRANCHLENGTH);


    bool uselayers=false;
    if(layer_branch.length())
        uselayers=true;
    else
        maxlayers=1;

    __hidden::indata layerbranch;
    if(uselayers)
        layerbranch.createFrom({layer_branch}, {1.}, {0.}, MAXBRANCHLENGTH);

    __hidden::indata xy;
    xy.createFrom({xbranch, ybranch}, {1., 1.}, {0., 0.}, MAXBRANCHLENGTH);
    bool XisPhi=branchIsPhi(xbranch);
    bool YisPhi=branchIsPhi(ybranch);

    __hidden::indata xy_center;
    xy_center.createFrom({xcenter, ycenter}, {1., 1.}, {0., 0.}, 1);

    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);

    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get(treename);


    //the order is important!
    branch.setup(tree);
    if(uselayers)
        layerbranch.setup(tree);
    //
    xy.setup(tree);
    xy_center.setup(tree);
    counter.setup(tree);

    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
    for(int jet=0;jet<nevents;jet++){


        branch.zeroAndGet(jet);
        if(uselayers)
            layerbranch.zeroAndGet(jet);

        xy.zeroAndGet(jet);
        xy_center.zeroAndGet(jet);
        counter.zeroAndGet(jet);

        std::vector<std::vector<std::vector<float> > >
        entriesperpixel(xbins,std::vector<std::vector<float> >(ybins,std::vector<float>(maxlayers,0)));

        double xcentre=xy_center.getData(0, 0);
        double ycentre=xy_center.getData(1, 0);
        int ncharged = counter.getData(0, 0);

        for(size_t elem=0; elem < ncharged; elem++) {
            int xidx = square_bins(xy.getData(0, elem), xcentre, xbins, xwidth,XisPhi);
            int yidx = square_bins(xy.getData(1, elem), ycentre, ybins, ywidth,YisPhi);

            if(xidx == -1 || yidx == -1) continue;

            int layer=0;
            if(uselayers)
                layer=round(layerbranch.getData(0, elem))-layer_offset;

            if(layer>=maxlayers)
                layer=maxlayers-1;
            if(layer<0)
                layer=0;
            for(size_t i_feat=0;i_feat<branch.nfeatures();i_feat++){

                float featval=0;
                if(maskedlayerbranch>=0 && (size_t)maskedlayerbranch==i_feat)
                    featval=(float)layer/s_norms.at(i_feat);
                else
                    featval=branch.getData(i_feat, elem);

                if(fillmodes.at(i_feat) == fm_single)
                    numpyarray[jet][xidx][yidx][layer][i_feat]=featval;
                else if(fillmodes.at(i_feat) == fm_relXsingle)
                    numpyarray[jet][xidx][yidx][layer][i_feat]=featval-xcentre;
                else if(fillmodes.at(i_feat) == fm_relYsingle)
                    numpyarray[jet][xidx][yidx][layer][i_feat]=featval-ycentre;
                else //(fillmodes.at(i_feat)==fm_sum || fillmodes.at(i_feat)==fm_average)
                    numpyarray[jet][xidx][yidx][layer][i_feat]+=featval;

            }

            entriesperpixel.at(xidx).at(yidx).at(layer)++;
        }


        //average
        for(size_t i_feat=0;i_feat<branch.nfeatures();i_feat++){
            if(fillmodes.at(i_feat)!=fm_average)
                continue;
            for(int i=0;i<xbins;i++){
                for(int j=0;j<ybins;j++){
                    for(int l=0;l<maxlayers;l++){
                        if(entriesperpixel.at(i).at(j).at(l))
                            numpyarray[jet][i][j][l][i_feat] /= entriesperpixel[i][j][l];
                    }
                }
            }
        }

    }

}

void priv_fillDensityMap(boost::python::numeric::array numpyarray,
        double norm,
        std::string in_branch,
        std::string in_weightbranch,
        std::string filename, std::string counter_branch,
        std::string xbranch, std::string xcenter, int xbins, float xwidth,
        std::string ybranch, std::string ycenter, int ybins, float ywidth,
        double offset, bool count
){


    TString branchstr=in_branch;
    TString weightstr=in_weightbranch;

    __hidden::indata branch;
    if(!count)
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
    bool XisPhi=branchIsPhi(xbranch);
    bool YisPhi=branchIsPhi(ybranch);

    //get jet center
    __hidden::indata xy_center;
    xy_center.createFrom({xcenter, ycenter}, {1., 1.}, {0., 0.}, 1);

    //get particle counter
    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);

    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get(treename);

    //connect all branches
    if(!count)
        branch.setup(tree);
    if(useweights)
        weight.setup(tree);
    xy.setup(tree);
    xy_center.setup(tree);
    counter.setup(tree);

    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
    for(int jet=0;jet<nevents;jet++){
        if(!count)
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
            int xidx = square_bins(xy.getData(0, elem), xy_center.getData(0, 0), xbins, xwidth,XisPhi);
            int yidx = square_bins(xy.getData(1, elem), xy_center.getData(1, 0), ybins, ywidth,YisPhi);
            if(xidx == -1 || yidx == -1) continue;
            float feature_value = 1;
            if(!count)
                feature_value = branch.getData(0, elem)-offset;
            //std::cout << elem << xidx << "  " << yidx << " "<< feature_value <<std::endl;
            float weight_value =1;
            if(useweights)
                weight_value = weight.getData(0, elem);

            densemap.at(xidx).at(yidx)+=feature_value*weight_value;

        }

        //seems to need burte-force way because of boost:numpy
        for(int i=0;i<xbins;i++){
            for(int j=0;j<ybins;j++){
                numpyarray[jet][i][j][0]=densemap.at(i).at(j);
            }
        }


    }

    tfile->Close();
    delete tfile;
}

void zeroPad() {
	//make real zero pad
	__hidden::indata::meanPadding = false;
}

void meanPad() {
	//make real zero pad
	__hidden::indata::meanPadding = true;
}
void setTreeName(std::string name){
    treename=name;
}
void doScaling(bool doit){
    __hidden::indata::doscaling=doit;
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
    def("fillCountMap", &fillCountMap);
    def("fillDensityLayers", &fillDensityLayers);
    def("meanPad", &meanPad);
    def("zeroPad", &zeroPad);
    def("setTreeName", &setTreeName);
    def("doScaling", &doScaling);
}
