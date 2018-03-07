/*
 * helper.h
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_HELPER_H_
#define DEEPJET_MODULES_INTERFACE_HELPER_H_


#include <dirent.h>
#include <stdlib.h>
#include "TString.h"

TString prependXRootD(const TString& path);

bool isApprox(const float& a , const float& b, float eps=0.001);

float deltaPhi(const float& phi1, const float& phi2);

#endif /* DEEPJET_MODULES_INTERFACE_HELPER_H_ */
