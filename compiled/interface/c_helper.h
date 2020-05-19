/*
 * helper.h
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_CHELPER_H_
#define DEEPJET_MODULES_INTERFACE_CHELPER_H_


#include <dirent.h>
#include <stdlib.h>
#include "TString.h"
#include "TObject.h"
#include "TString.h"
#include <sstream>
#include <string>
#include <iostream>

TString prependXRootD(const TString& path);

bool isApprox(const float& a , const float& b, float eps=0.001);

float deltaPhi(const float& phi1, const float& phi2);

void checkTObject(const TObject * o, TString msg);

template<class T>
T*  getLineDouble(const T * h);

template <class T>
std::string to_str(const T& t){
    std::stringstream ss;
    ss << t;
    return ss.str();
}

template <class T>
std::string to_str(const std::vector<T>& t){
    std::stringstream ss;
    ss << "[";
    for(const auto& v:t)
        ss << " " << to_str(v);
    ss << " ]";
    return ss.str();
}



template<class T>
T*  getLineDouble(const T * h){
    T* h2 = (T*)h->Clone(h->GetName()+(TString)"dline");
    h2->SetLineWidth(h->GetLineWidth()+1);
    h2->SetLineColor(kBlack);
    h2->SetLineColorAlpha(kBlack,0.8);
    return h2;
}





#endif /* DEEPJET_MODULES_INTERFACE_HELPER_H_ */
