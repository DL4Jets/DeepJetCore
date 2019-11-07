/*
 * helper.cpp
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */


#include "../interface/helper.h"
#include <stdexcept>

namespace p = boost::python;
namespace np = boost::python::numpy;


np::ndarray simpleArrayToNumpy( djc::simpleArray<float>& ifarr){

    auto size = ifarr.size();
    auto shape =  ifarr.shape();
    for(const auto& s:shape){
        if(s<0)
            throw std::runtime_error("simpleArrayToNumpy: no conversion from ragged simpleArrys possible");
    }

    p::list pshape;
    for(const auto& s:shape)
        pshape.append(s);

    p::tuple tshape(pshape);//not working

    np::ndarray nparr = np::from_data((void*)ifarr.disownData(),
            np::dtype::get_builtin<float>(),
            p::make_tuple(size), p::make_tuple(sizeof(float)), p::object() );

    nparr = nparr.reshape(tshape);
    return nparr;

}


TString prependXRootD(const TString& path){

    TString full_path = realpath(path, NULL);
    if(full_path.BeginsWith("/eos/cms/")){
        TString append="root://eoscms.cern.ch//";
        TString s_remove="/eos/cms/";
        TString newpath (full_path(s_remove.Length(),full_path.Length()));
        newpath=append+newpath;
        return newpath;
    }
    return path;
}
bool isApprox(const float& a , const float& b, float eps){
    return fabs(a-b)<eps;
}

float deltaPhi(const float& a, const float& b){
    const float pi = 3.14159265358979323846;
    float delta = (a -b);
    while (delta >= pi)  delta-= 2* pi;
    while (delta < -pi)  delta+= 2* pi;
    return delta;
}



void checkTObject(const TObject* o, TString msg){
    TString mesg = msg;
    mesg += ": " ;
    const char * name  = o->GetName();
    mesg += (TString)name;
    if(!o || o->IsZombie()){
        throw std::runtime_error(mesg.Data());
    }



}
