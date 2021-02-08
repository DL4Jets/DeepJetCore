/*
 * version.h
 *
 *  Created on: 6 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_VERSION_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_VERSION_H_

#define DJCDATAVERSION (2.1f)
#define DJCDATAVERSION_COMPAT (2.0f)

bool checkVersionCompatible(const float& version);

inline bool checkVersionStrict(float version){
    return version == DJCDATAVERSION;
}


#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_VERSION_H_ */
