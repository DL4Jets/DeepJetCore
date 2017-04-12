/*
 * colorToTColor.h
 *
 *  Created on: 24 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_ANALYSIS_INTERFACE_COLORTOTCOLOR_H_
#define DEEPJET_ANALYSIS_INTERFACE_COLORTOTCOLOR_H_

#include "TString.h"
#include "TColor.h"
#include <iostream>
/*
 * black
 * gray
 * white
 * red
 * darkred
 * blue
 * darkblue
 * green
 * darkgreen
 * purple
 * darkpurple
 *
 */

int colorToTColor(const TString& str){
	TString copstr=str;
	copstr.ToLower();

	if(copstr=="black")
		return kBlack;

	if(copstr=="gray")
		return kGray+1;

	if(copstr=="white")
		return kWhite;

	if(copstr=="red")
		return kRed;

	if(copstr=="darkred")
		return kRed+2;

	if(copstr=="blue")
		return kAzure+2;

	if(copstr=="darkblue")
		return kBlue+1;

	if(copstr=="green")
		return kGreen+1;

	if(copstr=="darkgreen")
		return kGreen+3;

	if(copstr=="purple")
		return kMagenta+1;

	if(copstr=="darkpurple")
		return kMagenta+3;

	std::cerr<<"color string " << str << " not recognised\noptions"
			<<"\n\
 * black\n\
 * gray\n\
 * white\n\
 * red\n\
 * darkred\n\
 * blue\n\
 * darkblue\n\
 * green\n\
 * darkgreen\n\
 * purple\n\
 * darkpurple"<<std::endl;
	return kBlack;

}


#endif /* DEEPJET_ANALYSIS_INTERFACE_COLORTOTCOLOR_H_ */
