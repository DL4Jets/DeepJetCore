/*
 * version.cpp
 *
 *  Created on: 8 Feb 2021
 *      Author: jkiesele
 */

#include "../interface/version.h"
#include <iostream>

bool warning_issued=false;

bool checkVersionCompatible(const float& version){
    bool compatprevious = version == DJCDATAVERSION_COMPAT;
    if(compatprevious && !warning_issued){
        std::cout
        << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
        << "WARNING:\n"
        << "You are reading an old DeepJetCore file format ("<<DJCDATAVERSION_COMPAT <<") "
        << "while the current file format version is "<< DJCDATAVERSION <<".\n"
        << "The data can be read in compatibility mode,\n"
        << "but please update the data set by either recreating it\n"
        << "or using the script convertDCFromPreviousMinorVersion.py.\n"
        << "COMPATIBILITY WILL BE PROVIDED ONLY UNTIL THE NEXT FILE FORMAT CHANGE.\n"
        << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<< std::endl;
        warning_issued=true;
    }
    return version == DJCDATAVERSION || compatprevious;
}


