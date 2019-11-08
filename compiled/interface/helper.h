/*
 * helper.h
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_HELPER_H_
#define DEEPJET_MODULES_INTERFACE_HELPER_H_


#include <boost/python.hpp>
#include "boost/python/numpy.hpp"

#include <dirent.h>
#include <stdlib.h>
#include "TString.h"
#include "TObject.h"
#include "TString.h"
#include "simpleArray.h"

TString prependXRootD(const TString& path);

bool isApprox(const float& a , const float& b, float eps=0.001);

float deltaPhi(const float& phi1, const float& phi2);

void checkTObject(const TObject * o, TString msg);

/**
 * transfers ownership of the data to numpy array - no copy.
 */
boost::python::numpy::ndarray simpleArrayToNumpy( djc::simpleArray<float>& ifarr);

#endif /* DEEPJET_MODULES_INTERFACE_HELPER_H_ */
