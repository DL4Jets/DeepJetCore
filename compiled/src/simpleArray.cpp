#include "../interface/simpleArray.h"
#include <vector>


namespace djc{
/**
    * Split indices can directly be used with the split() function.
    * Returns e.g. {2,5,3,2}, which corresponds to DataSplitIndices of {2,7,10,12}
    */

std::vector<size_t>  simpleArrayBase::getSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
        bool sqelementslimit, bool strict_limit, std::vector<bool>& size_ok, std::vector<size_t>& nelemtns_per_split){
    return priv_getSplitIndices(false, rowsplits, nelements_limit, sqelementslimit,  size_ok, nelemtns_per_split,strict_limit);
}

/**
 * Split indices can directly be used with the split() function.
 * Returns row splits e.g. {2,7,10,12} which corresponds to Split indices of {2,5,3,2}
 */


std::vector<size_t>  simpleArrayBase::getDataSplitIndices(const std::vector<int64_t> & rowsplits, size_t nelements_limit,
        bool sqelementslimit, bool strict_limit, std::vector<bool>& size_ok, std::vector<size_t>& nelemtns_per_split){
    return priv_getSplitIndices(true, rowsplits, nelements_limit, sqelementslimit,  size_ok, nelemtns_per_split,strict_limit);
}

/**
 * Transforms row splits to n_elements per ragged sample
 */

std::vector<int64_t>  simpleArrayBase::dataSplitToSplitIndices(const std::vector<int64_t>& row_splits){
    if(!row_splits.size())
        throw std::runtime_error("simpleArrayBase::dataSplitToSplitIndices: row splits empty");
    auto out = std::vector<int64_t>(row_splits.size()-1);
    for(size_t i=0;i<out.size();i++){
        out.at(i) = row_splits.at(i+1)-row_splits.at(i);
    }
    return out;
}

/**
 * Transforms n_elements per ragged sample to row splits
 */

std::vector<int64_t>  simpleArrayBase::splitToDataSplitIndices(const std::vector<int64_t>& n_elements){
    auto out = std::vector<int64_t>(n_elements.size()+1);
    out.at(0)=0;
    int64_t last=0;
    for(size_t i=0;i<n_elements.size();i++){
        out.at(i+1) = last + n_elements.at(i);
        last = out.at(i+1);
    }
    return out;
}

simpleArrayBase::simpleArrayBase(std::vector<int> shape,const std::vector<int64_t>& rowsplits) {
    assigned_=false;
    shape_ = shape;
    if(rowsplits.size()){
        if(rowsplits.size() != shape_.at(0)+1)
            throw std::runtime_error("simpleArrayBase::simpleArrayBase: rowsplits.size() must equal shape[0] + 1");

        rowsplits_=rowsplits;
        shape_ = shapeFromRowsplits();
    }
    size_ = sizeFromShape(shape_);
}



std::string simpleArrayBase::dtypeToString(dtypes t){
    if(t==float32)
        return "float32";
    else if(t==int32)
        return "int32";
    else
        return "undef";
}
simpleArrayBase::dtypes simpleArrayBase::stringToDtype(const std::string& s){
    if(s=="float32")
        return float32;
    else if(s=="int32")
        return int32;
    else
        throw std::runtime_error("simpleArrayBase::dtypes simpleArrayBase::stringToDtype unknown dtype");
}

boost::python::list simpleArrayBase::shapePy()const{
    boost::python::list l;
    for(const auto& s: shape_)
        l.append(s);
    return l;
}

std::string simpleArrayBase::readDtypeFromFileP(FILE *& ifile)const{
    return dtypeToString(readDtypeTypeFromFileP(ifile));
}

std::string simpleArrayBase::readDtypeFromFile(const std::string& f)const{
    return dtypeToString(readDtypeTypeFromFile(f));
}

simpleArrayBase::dtypes simpleArrayBase::readDtypeTypeFromFileP(FILE *& ifile)const{
    long pos = ftell(ifile);

    float version = 0;
    io::readFromFile(&version, ifile);
    if(!checkVersionCompatible(version)){//compat
            throw std::runtime_error("simpleArrayBase::readDtypeTypeFromFileP: wrong format version");
    }
    dtypes dt=float32;
    if(checkVersionStrict(version))
        io::readFromFile(&dt, ifile);
    fseek(ifile,pos-ftell(ifile),SEEK_CUR);//go back
    return dt;
}

simpleArrayBase::dtypes simpleArrayBase::readDtypeTypeFromFile(const std::string& f)const{
    FILE *ifile = fopen(f.data(), "rb");
    if(!ifile)
        throw std::runtime_error("simpleArrayBase::readDtypeTypeFromFile: file "+f+" could not be opened.");
    float version = 0;
    io::readFromFile(&version, ifile);
    if(!checkVersionCompatible(version))
        throw std::runtime_error("simpleArrayBase::readDtypeTypeFromFile: wrong format version");
    auto type = readDtypeTypeFromFileP(ifile);
    fclose(ifile);
    return type;
}


std::vector<int64_t> simpleArrayBase::readRowSplitsFromFileP(FILE *& ifile, bool seeknext){

    float version = 0;
    size_t size;
    std::vector<int> shape;
    std::vector<int64_t> rowsplits;
    io::readFromFile(&version, ifile);
    if(!checkVersionCompatible(version))
        throw std::runtime_error("simpleArrayBase::readRowSplitsFromFileP: wrong format version");
    if(checkVersionStrict(version)){
        dtypes rdtype;
        std::string namedummy;
        std::vector<std::string> featnamedummy;
        io::readFromFile(&rdtype, ifile);
        io::readFromFile(&namedummy, ifile);
        io::readFromFile(&featnamedummy, ifile);
    }

    io::readFromFile(&size, ifile);

    size_t shapesize = 0;
    io::readFromFile(&shapesize, ifile);
    shape = std::vector<int>(shapesize, 0);
    io::readFromFile(&shape[0], ifile, shapesize);

    size_t rssize = 0;
    io::readFromFile(&rssize, ifile);
    rowsplits = std::vector<int64_t>(rssize, 0);

    if(rssize){
        quicklz<int64_t> iqlz;
        iqlz.readAll(ifile, &rowsplits[0]);
    }
    if(seeknext){
        quicklz<float> qlz;//template arg does not matter here
        qlz.skipBlock(ifile);//sets file point to next item
    }
    return rowsplits;
}


void simpleArrayBase::skipToNextArray(FILE *& ofile)const{
    readRowSplitsFromFileP(ofile,true);
}


std::vector<int64_t> simpleArrayBase::mergeRowSplits(const std::vector<int64_t> & rowsplitsa, const std::vector<int64_t> & rowsplitsb){
    if(rowsplitsb.size()<1)
        return rowsplitsa;
    if(rowsplitsa.size()<1)
        return rowsplitsb;
    std::vector<int64_t> out=rowsplitsa;
    out.resize(out.size() + rowsplitsb.size()-1);
    int64_t lasta = rowsplitsa.at(rowsplitsa.size()-1);

    for(size_t i=0;i<rowsplitsb.size();i++)
        out.at(i + rowsplitsa.size() - 1) = lasta + rowsplitsb.at(i);

    return out;
}


std::vector<int64_t> simpleArrayBase::splitRowSplits(std::vector<int64_t> & rowsplits, const size_t& splitpoint){

    if(splitpoint >= rowsplits.size())
        throw std::out_of_range("simpleArrayBase::splitRowSplits: split index out of range");

    int64_t rsatsplitpoint = rowsplits.at(splitpoint);
    std::vector<int64_t> out = std::vector<int64_t> (rowsplits.begin(),rowsplits.begin()+splitpoint+1);
    std::vector<int64_t> rhs = std::vector<int64_t>(rowsplits.size()-splitpoint);
    for(size_t i=0;i<rhs.size();i++)
        rhs.at(i) = rowsplits.at(splitpoint+i) - rsatsplitpoint;

    rowsplits = rhs;
    return out;
}


 std::vector<size_t>  simpleArrayBase::priv_getSplitIndices(bool datasplit, const std::vector<int64_t> & rowsplits, size_t nelements_limit,
        bool sqelementslimit, std::vector<bool>& size_ok, std::vector<size_t>& nelemtns_per_split, bool strict_limit){

    std::vector<size_t> outIdxs;
    size_ok.clear();
    nelemtns_per_split.clear();
    if(rowsplits.size()<1)
        return outIdxs;

    size_t i_old=0;
    size_t s_old = 0;
    size_t i_s = 0;
    while (true) {

        size_t s = rowsplits.at(i_s);
        size_t delta = s - s_old;
        size_t i_splitat = rowsplits.size()+1;

        if (sqelementslimit)
            delta *= delta;

        if (delta > nelements_limit && i_s != i_old+1) {
            i_splitat = i_s - 1;
            i_s--;
        }
        else if (delta == nelements_limit ||
                i_s == rowsplits.size() - 1 ||
                (delta > nelements_limit && i_s == i_old+1)) {
            i_splitat = i_s;
        }


        if (i_splitat < rowsplits.size() ) {        //split

            if(i_splitat==i_old){
                //sanity check, should not happen
                std::cout <<"simpleArrayBase::priv_getSplitIndices: attempting empty split at " << i_splitat << std::endl;
                throw std::runtime_error("simpleArrayBase::priv_getSplitIndices: attempting empty split");
            }


            size_t nelements = rowsplits.at(i_splitat) - rowsplits.at(i_old);
            bool is_good = (!strict_limit || nelements <= nelements_limit) && nelements>0;//safety for zero element splits
            size_ok.push_back(is_good);
            nelemtns_per_split.push_back(nelements);

            if(datasplit)
                outIdxs.push_back(i_splitat);
            else
                outIdxs.push_back(i_splitat - i_old);


            //std::cout << "i_old " << i_old << "\n";
            //std::cout << "i_s " << i_s << "\n";
            //std::cout << "s_old " << s_old << "\n";
            //std::cout << "s " << s << "\n";
            //std::cout << "i_splitat " << i_splitat << "\n";
            //std::cout << "is_good " << is_good << "\n";
            //std::cout << "i_splitat - i_old " << i_splitat - i_old << "\n";
            //std::cout << std::endl;

            i_old = i_splitat;
            s_old = rowsplits.at(i_old);
            //i_s = i_splitat;

        }
        i_s++;
        if(i_s >= rowsplits.size())
            break;
    }

    return outIdxs;
}


void simpleArrayBase::getFlatSplitPoints(size_t splitindex_begin, size_t splitindex_end,
        size_t & splitpoint_start, size_t & splitpoint_end)const{
    splitpoint_start = splitindex_begin;
    splitpoint_end = splitindex_end;
    if(isRagged()){
        splitpoint_start = rowsplits_.at(splitindex_begin);
        splitpoint_end = rowsplits_.at(splitindex_end);
        for (size_t i = 2; i < shape_.size(); i++){
            splitpoint_start *= (size_t)std::abs(shape_.at(i));
            splitpoint_end   *= (size_t)std::abs(shape_.at(i));
        }
    }
    else{
        for (size_t i = 1; i < shape_.size(); i++){
            splitpoint_start *= (size_t)std::abs(shape_.at(i));
            splitpoint_end   *= (size_t)std::abs(shape_.at(i));
        }
    }
}



size_t simpleArrayBase::sizeFromShape(const std::vector<int>& shape) const {
    int64_t size = 1;
    for (const auto s : shape){
        size *= std::abs(s);
        if(s<0)
            size=std::abs(s);//first ragged dimension has the full size of previous dimensions
    }
    return size;
}


std::vector<int> simpleArrayBase::shapeFromRowsplits()const{
    if(!isRagged()) return shape_;
    if(shape_.size()<2) return shape_;
    auto outshape = shape_;
    //rowsplits.size = nbatch+1
    outshape.at(1) = - (int)rowsplits_.at(rowsplits_.size()-1);
    return outshape;
}


void simpleArrayBase::checkShape(size_t ndims)const{
    //rowsplit ready due to definiton of shape
    if(ndims != shape_.size()){
        throw std::out_of_range("simpleArrayBase::checkShape: shape does not match dimensions accessed ("+
                std::to_string(ndims)+"/"+std::to_string(shape_.size())+") "+name_);
    }
}


void simpleArrayBase::checkSize(size_t idx)const{
    if(idx >= size_)
        throw std::out_of_range("simpleArrayBase::checkSize: index out of range");
}


void simpleArrayBase::checkRaggedIndex(size_t i, size_t j)const{
    if(i > rowsplits_.size()-1 || j >= rowsplits_.at(i+1)-rowsplits_.at(i))
        throw std::out_of_range("simpleArrayBase::checkRaggedIndex: index out of range");
}


template<>
simpleArrayBase::dtypes simpleArray_float32::dtype()const{return float32;}
template<>
simpleArrayBase::dtypes simpleArray_int32::dtype()const{return int32;}

}//ns
