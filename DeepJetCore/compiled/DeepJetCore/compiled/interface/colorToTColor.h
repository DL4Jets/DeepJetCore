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


int lineToTLineStyle(const TString& str){
    TString copstr=str;
    copstr.ToLower();
    if(copstr.Contains("solid"))
        return 1;

    if(copstr.Contains("dashed"))
        return 2;

    if(copstr.Contains("dotted"))
        return 3;

    return 1;
}


int colorToTColor(const TString& str){
    TString copstr=str;
    copstr.ToLower();

    if(copstr.Contains("black"))
        return kBlack;

    if(copstr.Contains("gray"))
        return kGray+1;

    if(copstr.Contains("white"))
        return kWhite;

    if(copstr.Contains("darkred"))
        return kRed+2;

    if(copstr.Contains("red"))
        return kRed;

    if(copstr.Contains("darkblue"))
        return kBlue+1;

    if(copstr.Contains("blue"))
        return kAzure+2;

    if(copstr.Contains("darkgreen"))
        return kGreen+3;

    if(copstr.Contains("green"))
        return kGreen+1;

    if(copstr.Contains("darkpurple"))
        return kMagenta+3;

    if(copstr.Contains("purple"))
        return kMagenta+1;


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
