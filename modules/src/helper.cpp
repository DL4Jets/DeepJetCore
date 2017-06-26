/*
 * helper.cpp
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */


#include "../interface/helper.h"


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
