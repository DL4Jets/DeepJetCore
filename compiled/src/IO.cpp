#include "../interface/IO.h"


namespace djc{
namespace io{

template <>
void writeToFile<std::string>(const std::string * p, FILE * ofile, size_t N, size_t Nbytes){
    N=p->length();
    Nbytes = N*sizeof(char);
    writeToFile<size_t>(&N,ofile);
    if(!N)
        return;
    size_t ret = fwrite(p->data(), 1, Nbytes, ofile);
    if(ret != Nbytes){
        std::string fname = followFileName(ofile);
        fclose(ofile);
        throw std::runtime_error("djc::io::writeToFile: writing to file "+fname+" not successful");
    }
}


template <>
void readFromFile<std::string>(std::string * p, FILE* ifile, size_t N, size_t Nbytes){

    readFromFile<size_t>(&N,ifile);
    if(!N){
        *p="";
        return;
    }
    char * c = new char[N];

    Nbytes = N* sizeof(char);
    size_t ret = fread(c, 1, Nbytes, ifile);
    *p = std::string(c,N);
    delete[] c;

    if(ret != Nbytes){
        std::string fname = followFileName(ifile);
        fclose(ifile);
        throw std::runtime_error("djc::io::readFromFile:reading from file "+fname+" not successful");
    }
}

}//ns
}//ns
