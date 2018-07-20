/*
 * helper.cpp
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */


#include "helper.h"

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
