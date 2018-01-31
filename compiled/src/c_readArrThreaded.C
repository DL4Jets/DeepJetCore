

#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "../interface/pythonToSTL.h"
#include "../interface/helper.h"

#include <iostream>

#include <pthread.h>


#include <stdio.h>
#include <stdlib.h>

#include "../interface/quicklz.h"

#define MAXCHUNK (0xffffffff - 400)

bool debug=false;

class readThread{
public:
    readThread(long arrpointer,
            const std::string& filenamein,
            long size,bool rmwhendone){

        arrbuf=(float*)(void*)arrpointer;
        infile=filenamein;
        length=size;
        length*=sizeof(float);
        pthread=new pthread_t();
        done=0;
        id=lastid;
        lastid++;
        if(lastid>0xFFFE)
            lastid=0;
        removewhendone=rmwhendone;
        state_decompress = new qlz_state_decompress();
    }

    ~readThread(){
        if(pthread)delete pthread;
        if(state_decompress)delete state_decompress;
    }
    void start(){
        int  iret= pthread_create( pthread, NULL, readArrThread, (void*) this);
        if(iret){
            std::cerr << "Error - pthread_create() return code: "<<iret <<std::endl;
            throw std::runtime_error("Unable to create thread");
        }
    }

    bool isDone()const{
        return done;
    }

    void readBlocking(){
        start();
        join();
    }

    void join(double timeout=0){
        pthread_join( *pthread, NULL);
    }

    static void * readArrThread( void *ptr );

    int getId()const{return id;}

private:
    static int lastid;
    float * arrbuf;
    std::string infile;
    size_t length;
    pthread_t * pthread;
    uint8_t done;
    int id;
    bool removewhendone;
    qlz_state_decompress *state_decompress;
};


int readThread::lastid=0;




void * readThread::readArrThread( void *ptr ){

    readThread* thisthread=(readThread*)ptr;
    if(debug)
        std::cout << "thread started " << thisthread->infile <<std::endl;
    FILE *ifile;
    uint8_t nchunks=0;

    ifile = fopen(thisthread->infile.data(), "rb");
    fseek(ifile, 0, SEEK_END);
    unsigned int filelength = ftell(ifile);
    fseek(ifile, 0, SEEK_SET);
    fread(&nchunks, 1, 1, ifile);
    std::vector<size_t> chunksizes(nchunks,0);
    size_t vecbytesize=nchunks*sizeof(size_t);
    fread(&chunksizes[0], 1, vecbytesize, ifile);

    if(debug){
        std::cout << "file has "<< (int)nchunks << " chunks"<<std::endl;
        for(auto v:chunksizes)
            std::cout << " "<< v;
        std::cout << std::endl;
    }

    //read in chunks
    size_t chunk=0;
    size_t totalsize=0;
    size_t writepos=0;
    size_t allread=0;
    char* src =0;
    if(debug)
        std::cout << "Full length " << thisthread->length << std::endl;
    while(chunk<nchunks){

        src = new char[chunksizes.at(chunk)];
        fread(src, 1, chunksizes.at(chunk), ifile);
        totalsize += qlz_size_decompressed(src);

        if(debug)
            std::cout << writepos << " " << totalsize << ", " << allread<<std::endl;

        //if(writepos)writepos--;

        allread += qlz_decompress(src, &(thisthread->arrbuf[writepos/sizeof(float)]), thisthread->state_decompress);
        if(debug)
            std::cout << "allread: " << allread << std::endl;
        writepos=totalsize;
        chunk++;
        delete src;
    }
    if(debug)
        std::cout << "allread "<< allread << " totalsize "<< totalsize <<std::endl;

    //while ... if len>thisthread->length throw

    //totalsize compare to vector

    if(allread!=thisthread->length){
        fclose(ifile);
        throw std::runtime_error("readArrThread:target array size does not match ");
    }
    fclose(ifile);
    thisthread->done=1;//atomic

    if(thisthread->removewhendone){//thisthread->infile.data()
        std::string rmstring="rm -f ";
        rmstring+=thisthread->infile;
        system(rmstring.data());
    }
    return 0;
}

using namespace boost::python;


//module info and interface

size_t maxreads=1000;
std::vector<readThread*> allreads(maxreads,0);
size_t acounter=0;

bool readBlocking(long arrpointer,
        std::string filenamein,
        const boost::python::list shape,
        bool rmwhendone){

    long length=1;
    std::vector<int> sshape=toSTLVector<int>(shape);
    for(const auto& s:sshape)
        length*=s;

    readThread * t=new readThread(arrpointer,filenamein,length,rmwhendone);
    t->readBlocking();
    bool succ=t->isDone();
    delete t;
    return succ;
}

int startReading(long arrpointer,
        std::string filenamein,
        const boost::python::list shape,
        bool rmwhendone){

    long length=1;
    std::vector<int> sshape=toSTLVector<int>(shape);
    for(const auto& s:sshape)
        length*=s;

    readThread * t=new readThread(arrpointer,filenamein,length,rmwhendone);
    t->start();
    if(allreads.at(acounter) && !allreads.at(acounter)->isDone())
        throw std::out_of_range("c_readArrThreaded::startReading: overflow. Increase number of maximum threads (setMax)");
    allreads.at(acounter)=t;
    acounter++;
    if(acounter>=maxreads)
        acounter=0;
    return t->getId();
}

bool isDone(int id){
    for(auto& t:allreads){
        if(!t)continue;
        if(t->getId()==id){
            if(t->isDone()){
                t->join();
                delete t;
                t=0;
                return true;
            }
            else{
                return false;
            }
        }
    }
    if(debug)
        std::cerr<<"isDone: ID "<< id << " not found "<<std::endl;
    return true;
}

void setMax(int m){
    if(m>0xFFFE)
        throw std::runtime_error("setMax: must be smaller than 65536");
    maxreads=m;
    allreads.resize(m,0);
}



void writeArray(long arrpointer,
        std::string file, const boost::python::list shape){

    long length=1;
    std::vector<int> sshape=toSTLVector<int>(shape);
    for(const auto& s:sshape)
        length*=s;

    length*=sizeof(float);

    FILE *ofile;
    char *src=(char*)(void*)arrpointer;
    char *dst;

    qlz_state_compress *state_compress = new qlz_state_compress();

    ofile = fopen(file.data(), "wb");

    // allocate "uncompressed size" + 400 for the destination buffer
    dst = new char [length + 400];

if(debug)
    std::cout << "array has "<< length << " bytes" <<std::endl;
    // compress and write result
    size_t remaininglength=length;
    size_t len2 =0;
    size_t startbyte=0;
    uint8_t nchunks=1;
    std::vector<size_t> chunksizes;

    while(remaininglength){

        size_t uselength=0;
        if(remaininglength > MAXCHUNK){
            uselength=MAXCHUNK;
            remaininglength-=MAXCHUNK;
            nchunks++;
            if(!nchunks){
                //throw etc
                //TBI (only kicks in at about 1TB)
            }

        }
        else{
            uselength=remaininglength;
            remaininglength=0;
        }
        size_t thissize = qlz_compress(&src[startbyte],&dst[len2], uselength, state_compress);
        chunksizes.push_back(thissize);
        len2+=thissize;
        startbyte+=uselength;
    }
    if(debug){
    std::cout << "writing "<< len2 << " compressed bytes in "<< (int)nchunks <<" chunks: " <<std::endl;
    for(const auto c:chunksizes)
        std::cout << c <<" ";
    std::cout << std::endl;
    }

    fwrite(&nchunks,1,1,ofile);
    fwrite(&chunksizes[0],1,chunksizes.size()*sizeof(size_t),ofile);
    fwrite(dst, len2, 1, ofile);


    fclose(ofile);

    delete dst;
    delete state_compress;

}


BOOST_PYTHON_MODULE(c_readArrThreaded) {
    //PyEval_InitThreads();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("readBlocking", &readBlocking);

    def("startReading", &startReading);
    def("isDone", &isDone);

    def("writeArray", &writeArray);

}
