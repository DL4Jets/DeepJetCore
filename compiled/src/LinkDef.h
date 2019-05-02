/*
 * classes.h
 *
 *  Created on: 30 Apr 2019
 *      Author: jkiesele
 */

#ifndef BIN_LINKDEF_H_
#define BIN_LINKDEF_H_

#ifdef __ROOTCLING__
#include <vector>

#pragma link C++ class std::vector<float> +;
#pragma link C++ class std::vector<std::vector<float> > +;
#pragma link C++ class std::vector<std::vector<std::vector<float> > > +;
#pragma link C++ class std::vector<std::vector<std::vector<std::vector<float> > > > +;

#endif

#endif /* BIN_LINKDEF_H_ */
