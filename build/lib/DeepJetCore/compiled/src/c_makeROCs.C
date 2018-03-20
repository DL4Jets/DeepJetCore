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
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "../interface/pythonToSTL.h"
#include "friendTreeInjector.h"
#include "rocCurveCollection.h"

using namespace boost::python; //for some reason....



void makeROCs(
        const boost::python::list intextfiles,
        const boost::python::list names,
        const boost::python::list probabilities ,
        const boost::python::list truths,
        const boost::python::list vetos,
        const boost::python::list colors,
        std::string outfile,
        const boost::python::list cuts,
        bool usecmsstyle,
        std::string firstcomment,
        std::string secondcomment,
        const boost::python::list invalidate,
        const boost::python::list extralegend,
        bool logy,
        bool individual,
        std::string xaxis,
        int nbins,
		std::string treename
) {


    std::vector<TString>  s_intextfiles=toSTLVector<TString>(intextfiles);
    std::vector<TString>  s_names = toSTLVector<TString>(names);
    std::vector<TString>  s_probabilities = toSTLVector<TString>(probabilities);
    std::vector<TString>  s_truths = toSTLVector<TString>(truths);
    std::vector<TString>  s_vetos = toSTLVector<TString>(vetos);
    std::vector<TString>  s_colors = toSTLVector<TString>(colors);
    std::vector<TString>  s_cuts = toSTLVector<TString>(cuts);
    std::vector<TString>  s_invalidate =toSTLVector<TString>(invalidate);
    std::vector<TString>  s_extralegend=toSTLVector<TString>(extralegend);
    /*
     * Size checks!!!
     */
    if(s_intextfiles.size() !=s_names.size()||
            s_names.size() != s_probabilities.size() ||
            s_names.size() != s_truths.size()||
            s_names.size() != s_vetos.size()||
            s_names.size() != s_colors.size()||
            s_names.size() != s_cuts.size() ||
            s_invalidate.size() != s_names.size())
        throw std::runtime_error("makeROCs: input lists must have same size");

    //make unique list of infiles
    std::vector<TString> u_infiles;
    std::vector<TString> aliases;
    TString oneinfile="";
    bool onlyonefile=true;
    for(const auto& f:s_intextfiles){
        if(oneinfile.Length()<1)
            oneinfile=f;
        else
            if(f!=oneinfile)
                onlyonefile=false;
    }
    for(const auto& f:s_intextfiles){
        //if(std::find(u_infiles.begin(),u_infiles.end(),f) == u_infiles.end()){
        u_infiles.push_back(f);
        TString s="";
        s+=aliases.size();
        aliases.push_back(s);
        //	std::cout << s <<std::endl;
        //}
    }



    friendTreeInjector injector((TString)treename);
    std::vector<friendTreeInjector> injectors(u_infiles.size());
    std::vector<TChain*> chains(u_infiles.size());
    if(individual){
        if(u_infiles.size() != s_names.size())
            throw std::runtime_error("makeROCs: file list must have same size as legends etc. in individual mode");
        for(size_t i=0;i<u_infiles.size();i++){
            injectors.at(i).addFromFile((TString)u_infiles.at(i));
            injectors.at(i).createChain();
            chains.at(i)=injectors.at(i).getChain();
        }

    }
    else{
        for(size_t i=0;i<u_infiles.size();i++){
            if(!aliases.size())
                injector.addFromFile((TString)u_infiles.at(i));
            else
                injector.addFromFile((TString)u_infiles.at(i),aliases.at(i));
        }
        injector.createChain();
    }

    TString xaxisstr=xaxis;

    rocCurveCollection rocs;
    rocs.setNBins(nbins);
    rocs.setXaxis(xaxisstr);

    rocs.setCommentLine0(firstcomment.data());
    rocs.setCommentLine1(secondcomment.data());
    rocs.setLogY(logy);
    rocs.setCMSStyle(usecmsstyle);

    for(size_t i=0;i<s_names.size();i++){
        if(s_cuts.size())
            rocs.addROC(s_names.at(i),s_probabilities.at(i),s_truths.at(i),
                    s_vetos.at(i),s_colors.at(i),s_cuts.at(i),s_invalidate.at(i));
        else
            rocs.addROC(s_names.at(i),s_probabilities.at(i),s_truths.at(i),
                    s_vetos.at(i),s_colors.at(i),"",s_invalidate.at(i));
    }
    for(const auto& s:s_extralegend)
        rocs.addExtraLegendEntry(s);

    if(individual){
        rocs.printRocs(0,(TString)outfile,"",0,0,&chains);
    }
    else{
        rocs.printRocs(injector.getChain(),(TString)outfile);
    }
}




// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_makeROCs) {
    //__hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
    //anyway, it doesn't hurt, just leave this here
    def("makeROCs", &makeROCs);

}
