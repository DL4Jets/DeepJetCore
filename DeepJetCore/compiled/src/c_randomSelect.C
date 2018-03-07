/*
 * c_randomSelect.C
 *
 *  Created on: 14 Aug 2017
 *      Author: jkiesele
 */




#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
#include <boost/python/exception_translator.hpp>
#include <numeric> 

#include <exception>
#include <random>
#include "TRandom3.h"

using namespace boost::python;

class randomSelector{
public:
    randomSelector(){
        rand_=new TRandom3();
    }
    ~randomSelector(){
        delete rand_;
    }
    void select(boost::python::numeric::array probs , boost::python::numeric::array indices, const size_t nselect);
private:
    TRandom3* rand_;
} sel;


void randomSelector::select(boost::python::numeric::array  probs, boost::python::numeric::array  selects, const size_t nselect){

    const size_t size = len(probs);
    if(nselect>size){
        throw std::logic_error("randomSelector::select: can't select more than given");
    }
    std::vector<size_t> indices(size,0) ;
    std::iota(std::begin(indices), std::end(indices), 0);
    std::shuffle(indices.begin(), indices.end(), std::random_device());//hardware random device

    size_t nselected=0;
    size_t i_it=0;
    while(true){
        size_t i=indices.at(i_it);

        float prob=rand_->Uniform(0,1);
        if(probs[i]<prob && !selects[i]){//not yet selected
            selects[i]=1;
            nselected++;
        }

        if(nselected>=nselect)break;
        i_it++;
        if(i_it>=size)i_it=0;
    }
}

//indices are initialised to 0, probs describe the remove probabilities
void randSelect(boost::python::numeric::array probs,
        boost::python::numeric::array indices,
        int nentries){

    sel.select(probs,indices,nentries);

}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_randomSelect) {
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("randSelect", &randSelect);
}

