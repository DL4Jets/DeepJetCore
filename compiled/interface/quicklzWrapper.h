/*
 * quicklzWrapper.h
 *
 *  Created on: 5 Nov 2019
 *      Author: jkiesele
 */

#ifndef DEEPJETCORE_COMPILED_INTERFACE_QUICKLZWRAPPER_H_
#define DEEPJETCORE_COMPILED_INTERFACE_QUICKLZWRAPPER_H_

#include "quicklz.h"
#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <string>
#include <stdexcept>
#include "IO.h"
#include "version.h"
#include <iostream>

#define QUICKLZ_MAXCHUNK (0xffffffff - 400)

namespace djc{
template <class T>
class quicklz{
public:

    quicklz();
    ~quicklz();

    void reset();

    //reads header, saves total uncompressed size
    void readHeader(FILE *& ifile);

    //get uncompressed size to allocate memory if needed
    //not in bytes but in terms of T
    size_t getSize()const{return totalbytes_/sizeof(T);}

    //writes from compressed file to memory
    //returns in terms of T how many elements have been read
    size_t readCompressedBlock(FILE *& ifile, T * arr);

    //assumes you know the size that is supposed to be read
    //and memory has been allocated already!
    //returns in terms of T how many compressed elements have been read (without header)
    size_t readAll(FILE *& ifile, T * arr);

    //skips over the next compressed block without reading it
    size_t skipBlock(FILE *& ifile);

    //writes header and compressed data
    //give size in terms of T
    void writeCompressed(const T * arr, size_t size, FILE *& ofile);


private:
    std::vector<size_t> chunksizes_;
    uint8_t nchunks_;
    size_t totalbytes_;
    qlz_state_decompress *state_decompress_;
    qlz_state_compress *state_compress_;
};

template <class T>
quicklz<T>::quicklz(){
    nchunks_=0;
    totalbytes_=0;
    state_decompress_ = new qlz_state_decompress();
    state_compress_ = new qlz_state_compress();
}


template <class T>
quicklz<T>::~quicklz(){
    delete state_decompress_;
    delete state_compress_ ;
}

template <class T>
void quicklz<T>::reset(){
    chunksizes_.clear();
    nchunks_ = 0;
    totalbytes_ = 0;
    delete state_decompress_;
    delete state_compress_;
    state_decompress_ = new qlz_state_decompress();
    state_compress_ = new qlz_state_compress();
}

template <class T>
void quicklz<T>::readHeader(FILE *& ifile) {
    nchunks_ = 0;
    chunksizes_.clear();
    totalbytes_ = 0;
    float version = 0;
    io::readFromFile(&version, ifile);
    if(!checkVersionCompatible(version))
        throw std::runtime_error("quicklz<T>::readHeader: incompatible version");
    io::readFromFile(&nchunks_,  ifile);
    chunksizes_ = std::vector<size_t>(nchunks_, 0);
    io::readFromFile(&chunksizes_[0], ifile, nchunks_);
    io::readFromFile(&totalbytes_, ifile);
}




template <class T>
size_t quicklz<T>::readCompressedBlock(FILE *& ifile, T * arr){

    size_t chunk = 0;
    size_t allread = 0;
    char* src = 0;
    char * dst = (char*)(void*)arr;

    while (chunk < nchunks_ && totalbytes_) {
        //std::cout << "chunk with size " << chunksizes_.at(chunk) <<" size of " << sizeof(T) <<" total bytes "<< totalbytes_ << std::endl;
        src = new char[chunksizes_.at(chunk)];
        io::readFromFile(src, ifile, 0, chunksizes_.at(chunk));
        size_t readbytes = qlz_size_decompressed(src);
        //std::cout << "bytes to be decompressed " << readbytes << std::endl;

        allread += qlz_decompress(src, dst, state_decompress_);
        //std::cout << "decompress success " << readbytes << " allread " << allread << std::endl;
        chunk++;
        dst += readbytes;
        delete src;
    }
    if (allread != totalbytes_) {
        std::string moreinfo = "\nexpected: ";
        moreinfo += std::to_string(totalbytes_);
        moreinfo += " got: ";
        moreinfo += std::to_string(allread);
        delete state_decompress_;
        state_decompress_ = 0;
        throw std::runtime_error((
                "quicklz::readCompressedBlock: expected size and uncompressed size don't match: "+moreinfo));
    }
    return allread / sizeof(T);
}



template<class T>
size_t quicklz<T>::readAll(FILE *& ifile, T * arr) {
    readHeader(ifile);
    return readCompressedBlock(ifile, arr);
}

template<class T>
size_t quicklz<T>::skipBlock(FILE *& ifile){
    readHeader(ifile);
    size_t totalbytescompressed = 0;
    for(const auto& c:chunksizes_)
        totalbytescompressed+=c;
    fseek(ifile,totalbytescompressed,SEEK_CUR);
    return totalbytescompressed;
}

template<class T>
void quicklz<T>::writeCompressed(const T * arr, size_t size, FILE *& ofile) {

    size_t length = size * sizeof(T);
    const char *src = (const char*) (const void*) arr;

    //destination buffer
    char *dst = new char[length + 400];
    size_t remaininglength = length;
    size_t len2 = 0;
    size_t startbyte = 0;
    uint8_t nchunks = 1;
    std::vector<size_t> chunksizes;

    while (remaininglength) {

        size_t uselength = 0;
        if (remaininglength > QUICKLZ_MAXCHUNK) {
            uselength = QUICKLZ_MAXCHUNK;
            remaininglength -= QUICKLZ_MAXCHUNK;
            nchunks++;
            if (!nchunks) {
                throw std::runtime_error(
                        "quicklz::writeCompressed: array size too big (O(TB))!");
            }

        } else {
            uselength = remaininglength;
            remaininglength = 0;
        }
        size_t thissize = qlz_compress(&src[startbyte], &dst[len2], uselength,
                state_compress_);
        chunksizes.push_back(thissize);
        len2 += thissize;
        startbyte += uselength;
    }
    float version = DJCDATAVERSION;
    io::writeToFile(&version,ofile);
    io::writeToFile(&nchunks,ofile);
    io::writeToFile(&chunksizes[0],ofile,chunksizes.size());
    io::writeToFile(&length, ofile);
    io::writeToFile(dst, ofile, 0, len2);

    //end
    delete dst;
}

}//namespace

#endif
