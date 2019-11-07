/*
 * IO.h
 *
 *  Created on: 7 Nov 2019
 *      Author: jkiesele
 */

#ifndef DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_IO_H_
#define DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_IO_H_

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdexcept>

/*
 * Very simple template wrapper around fread and fwrite with error checks
 * The number of datatypes written is NOT given in bytes.
 * Only works for types with valid sizeof(type).
 * Otherwise specify number of bytes
 */

namespace djc{
namespace io{
//only linux
inline std::string followFileName(FILE * ofile){
    char proclnk[0xFFF];
    char filename[0xFFF];
    sprintf(proclnk, "/proc/self/fd/%d", fileno(ofile));
    int r = readlink(proclnk, filename, 0xFFF);
    std::string fname="uknown";
    if(r>0){
        if(r>=0xFFF-1)
            r = 0xFFF-1;
        filename[r]='\0';
        fname=filename;
    }
    return fname;
}

template <class T>
void writeToFile(T * p, FILE * ofile, size_t N=1, size_t Nbytes=0){
    if(!Nbytes){
        Nbytes = N*sizeof(T);
    }
    size_t ret = fwrite(p, 1, Nbytes, ofile);
    if(ret != Nbytes){
        std::string fname = followFileName(ofile);
        fclose(ofile);
        throw std::runtime_error("writing to file "+fname+" not successful");
    }
}

template <class T>
void readFromFile(T * p, FILE* ifile, size_t N=1, size_t Nbytes=0){
    if(!Nbytes)
        Nbytes = N* sizeof(T);
    size_t ret = fread(p, 1, Nbytes, ifile);
    if(ret != Nbytes){
        std::string fname = followFileName(ifile);
        fclose(ifile);
        throw std::runtime_error("reading from file "+fname+" not successful");
    }
}
}
}

#endif /* DJCDEV_DEEPJETCORE_COMPILED_INTERFACE_IO_H_ */
