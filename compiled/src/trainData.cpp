#include "../interface/trainData.h"


namespace djc{

/*
 *
    std::vector<simpleArray_float32> farrs_;
    std::vector<simpleArray_int32> iarrs_;

    enum typesorting{isfloat,isint};
    std::vector<std::pair<typesorting,size_t> > sorting_;
 */

void typeContainer::push_back(simpleArrayBase& a){
    if(a.dtype() == simpleArrayBase::float32){
        farrs_.push_back(dynamic_cast<simpleArray_float32&>(a));
        sorting_.push_back({isfloat,farrs_.size()-1});
    }
    else {//if(a.dtype() == simpleArrayBase::int32){
        iarrs_.push_back(dynamic_cast<simpleArray_int32&>(a));
        sorting_.push_back({isint,iarrs_.size()-1});
    }
}
void typeContainer::move_back(simpleArrayBase& a){
    if(a.dtype() == simpleArrayBase::float32){
        farrs_.push_back(std::move(dynamic_cast<simpleArray_float32&>(a)));
        sorting_.push_back({isfloat,farrs_.size()-1});
    }
    else {//if(a.dtype() == simpleArrayBase::int32){
        iarrs_.push_back(std::move(dynamic_cast<simpleArray_int32&>(a)));
        sorting_.push_back({isint,iarrs_.size()-1});
    }
}
bool typeContainer::operator==(const typeContainer& rhs)const{
    if(size() != rhs.size())
        return false;
    if(farrs_.size() != rhs.farrs_.size())
        return false;

    if(sorting_ != rhs.sorting_)
        return false;

    for(size_t i=0;i<farrs_.size();i++){
        if(farrs_.at(i) != rhs.farrs_.at(i))
            return false;
    }
    for(size_t i=0;i<iarrs_.size();i++){
        if(iarrs_.at(i) != rhs.iarrs_.at(i))
            return false;
    }
    return true;
}
simpleArrayBase& typeContainer::at(size_t idx){
    if(idx>=sorting_.size())
        throw std::out_of_range("typeContainer::at: requested "+std::to_string(idx)+" of "+std::to_string(sorting_.size()));
    auto s = sorting_.at(idx);
    if(s.first == isfloat)
        return farrs_.at(s.second);
    else //if(s.first == isint)
        return iarrs_.at(s.second);

}
const simpleArrayBase& typeContainer::at(size_t idx)const{
    if(idx>=sorting_.size())
        throw std::out_of_range("typeContainer::at: requested "+std::to_string(idx)+" of "+std::to_string(sorting_.size()));
    auto s = sorting_.at(idx);
    if(s.first == isfloat)
        return farrs_.at(s.second);
    else //if(s.first == isint)
        return iarrs_.at(s.second);
}


simpleArray_float32& typeContainer::at_asfloat32(size_t idx){
    if(at(idx).dtype() != simpleArrayBase::float32)
        throw std::runtime_error("typeContainer::at_asfloat32: is not float32");
    return dynamic_cast<simpleArray_float32&>(at(idx));
}

const simpleArray_float32& typeContainer::at_asfloat32(size_t idx)const{
    if(at(idx).dtype() != simpleArrayBase::float32)
        throw std::runtime_error("typeContainer::at_asfloat32: is not float32");
    return dynamic_cast<const simpleArray_float32&>(at(idx));
}

simpleArray_int32& typeContainer::at_asint32(size_t idx){
    if(at(idx).dtype() != simpleArrayBase::int32)
        throw std::runtime_error("typeContainer::at_asfloat32: is not float32");
    return dynamic_cast<simpleArray_int32&>(at(idx));
}

const simpleArray_int32& typeContainer::at_asint32(size_t idx)const{
    if(at(idx).dtype() != simpleArrayBase::int32)
        throw std::runtime_error("typeContainer::at_asfloat32: is not float32");
    return dynamic_cast<const simpleArray_int32&>(at(idx));
}


void typeContainer::clear(){
    farrs_.clear();
    iarrs_.clear();
    sorting_.clear();
}

void typeContainer::writeToFile(FILE *& ofile) const{
    size_t isize=size();
    io::writeToFile(&isize,ofile);
    for(const auto& i: sorting_){
        if(i.first == isfloat){
            farrs_.at(i.second).addToFileP(ofile);
        }
        else {// if(i.first == isint){
            iarrs_.at(i.second).addToFileP(ofile);
        }
    }
}

void typeContainer::readFromFile_priv(FILE *& ifile, bool justmetadata){
    clear();
    size_t isize = 0;
    io::readFromFile(&isize,ifile);
    for(size_t i=0;i<isize;i++){
        simpleArray_float32 tmp;//type doesn't matter
        auto dtype = tmp.readDtypeTypeFromFileP(ifile);
        if(dtype == simpleArrayBase::float32){
            simpleArray_float32 farr;
            farr.readFromFileP(ifile,justmetadata);
            move_back(farr);
        }
        else{ //if(dtype==simpleArrayBase::int32){
            simpleArray_int32 iarr;
            iarr.readFromFileP(ifile,justmetadata);
            move_back(iarr);
        }
    }
}

////////////////// trainData //////////////////////

bool trainData::operator==(const trainData& rhs)const{

    if(feature_arrays_ != rhs.feature_arrays_)
        return false;
    if(truth_arrays_ != rhs.truth_arrays_)
        return false;
    if(weight_arrays_ != rhs.weight_arrays_)
        return false;
    if(feature_shapes_ != rhs.feature_shapes_)
        return false;
    if(truth_shapes_ != rhs.truth_shapes_)
        return false;
    if(weight_shapes_ != rhs. weight_shapes_)
        return false;
    return true;
}


int trainData::storeFeatureArray(simpleArrayBase & a){
    size_t idx = feature_arrays_.size();
    feature_arrays_.move_back(a);
    a.clear();
    updateShapes();
    return idx;
}

int trainData::storeTruthArray(simpleArrayBase& a){
    size_t idx = truth_arrays_.size();
    truth_arrays_.move_back(a);
    a.clear();
    updateShapes();
    return idx;
}

int trainData::storeWeightArray(simpleArrayBase & a){
    size_t idx = weight_arrays_.size();
    weight_arrays_.move_back(a);
    a.clear();
    updateShapes();
    return idx;
}

void trainData::truncate(size_t position){
    *this = split(position);
}


void trainData::append(const trainData& td) {
    //allow empty append
    if (!feature_arrays_.size() && !truth_arrays_.size()
            && !weight_arrays_.size()) {
        *this = td;
        return;
    }
    if(!td.feature_arrays_.size() && !td.truth_arrays_.size()
            && !td.weight_arrays_.size()){
        return ; //nothing to do
    }
    if (feature_arrays_.size() != td.feature_arrays_.size()
            || truth_arrays_.size() != td.truth_arrays_.size()
            || weight_arrays_.size() != td.weight_arrays_.size()) {
        std::cout << "nfeat " << feature_arrays_.size() << "-" << td.feature_arrays_.size() <<'\n'
                << "ntruth " << truth_arrays_.size() << "-" << td.truth_arrays_.size()<<'\n'
                << "nweights " << weight_arrays_.size() << "-" <<  td.weight_arrays_.size() <<std::endl;
        throw std::out_of_range("trainData<T>::append: format not compatible.");
    }
    for(size_t i=0;i<feature_arrays_.size();i++)
        feature_arrays_.at(i).append(td.feature_arrays_.at(i));
    for(size_t i=0;i<truth_arrays_.size();i++)
        truth_arrays_.at(i).append(td.truth_arrays_.at(i));
    for(size_t i=0;i<weight_arrays_.size();i++)
        weight_arrays_.at(i).append(td.weight_arrays_.at(i));
    updateShapes();
}


trainData trainData::split(size_t splitindex) {
    trainData out;

    std::vector<std::pair< typeContainer* , typeContainer*> > vv = {
            {&feature_arrays_, &out.feature_arrays_},
            {&truth_arrays_, &out.truth_arrays_},
            {&weight_arrays_, &out.weight_arrays_}};
    for(auto& a: vv){
        for (size_t i=0;i<a.first->size();i++){
            if(a.first->dtype(i) == simpleArrayBase::float32){
                auto split = a.first->at_asfloat32(i).split(splitindex);
                a.second->push_back(split);
            }
            else if(a.first->dtype(i) == simpleArrayBase::int32){
                auto split = a.first->at_asint32(i).split(splitindex);
                a.second->push_back(split);
            }
            else{
                throw std::runtime_error("trainData::split: do not understand dtype");
            }
        }
    }

    updateShapes();
    out.updateShapes();
    return out;
}

trainData trainData::getSlice(size_t splitindex_begin, size_t splitindex_end)const{
    trainData out;


    std::vector<std::pair<const typeContainer* , typeContainer*> > vv = {
            {&feature_arrays_, &out.feature_arrays_},
            {&truth_arrays_, &out.truth_arrays_},
            {&weight_arrays_, &out.weight_arrays_}};
    for(auto& a: vv){
        for (size_t i=0;i<a.first->size();i++){
            if(a.first->dtype(i) == simpleArrayBase::float32){
                auto split = a.first->at_asfloat32(i).getSlice(splitindex_begin,splitindex_end);
                a.second->push_back(split);
            }
            else if(a.first->dtype(i) == simpleArrayBase::int32){
                auto split = a.first->at_asint32(i).getSlice(splitindex_begin,splitindex_end);
                a.second->push_back(split);
            }
        }
    }

    out.updateShapes();
    return out;
}

trainData trainData::shuffle(const std::vector<size_t>& shuffle_idxs)const{
    trainData out;
    std::vector<std::pair<const typeContainer* , typeContainer*> > vv = {
            {&feature_arrays_, &out.feature_arrays_},
            {&truth_arrays_, &out.truth_arrays_},
            {&weight_arrays_, &out.weight_arrays_}};
    for(auto& a: vv){
        for (size_t i=0;i<a.first->size();i++){
            if(a.first->dtype(i) == simpleArrayBase::float32){
                auto split = a.first->at_asfloat32(i).shuffle(shuffle_idxs);
                a.second->push_back(split);
            }
            else if(a.first->dtype(i) == simpleArrayBase::int32){
                auto split = a.first->at_asint32(i).shuffle(shuffle_idxs);
                a.second->push_back(split);
            }
            else{
                throw std::runtime_error("trainData::shuffle: do not understnad dtype");
            }
        }
    }

    out.updateShapes();
    return out;

}

bool trainData::validSlice(size_t splitindex_begin, size_t splitindex_end)const{

    const std::vector<const typeContainer* > vv = {&feature_arrays_, &truth_arrays_, &weight_arrays_};
    for(const auto& a: vv)
        for (size_t i=0;i<a->size();i++)
            if(! a->at(i).validSlice(splitindex_begin,splitindex_end))
                return false;

    return true;
}

void trainData::writeToFile(std::string filename)const{

    FILE *ofile = fopen(filename.data(), "wb");
    addToFileP(ofile);
    fclose(ofile);

}

void trainData::addToFile(std::string filename)const{

    FILE *ofile = fopen(filename.data(), "ab");
    addToFileP(ofile);
    fclose(ofile);
}

void trainData::addToFileP(FILE *& ofile)const{
    float version = DJCDATAVERSION;
    io::writeToFile(&version, ofile);

    //shape infos only
    writeNested(getShapes(feature_arrays_), ofile);
    writeNested(getShapes(truth_arrays_), ofile);
    writeNested(getShapes(weight_arrays_), ofile);

    //data
    feature_arrays_.writeToFile(ofile);
    truth_arrays_.writeToFile(ofile);
    weight_arrays_.writeToFile(ofile);
}

void trainData::priv_readFromFile(std::string filename, bool memcp){
    clear();
    FILE *ifile = fopen(filename.data(), "rb");
    char *buf = 0;
    if(false && memcp){
        FILE *diskfile = ifile;
        //check if exists before trying to memcp.
        checkFile(ifile, filename); //not set at start but won't be used

        fseek(diskfile, 0, SEEK_END);
        size_t fsize = ftell(diskfile);
        fseek(diskfile, 0, SEEK_SET);  /* same as rewind(f); */

        buf = new char[fsize];
        int ret = fread(buf, 1, fsize, diskfile);
        if(!ret){
            delete buf;
            throw std::runtime_error("trainData<T>::readFromFile: could not read file in memcp mode");
        }
        fclose(diskfile);

        ifile = fmemopen(buf,fsize,"r");
    }

    priv_readSelfFromFileP(ifile,filename);
    //check for eof and add until done. the append step can be heavily optimized! FIXME
    //read one more byte
    int ch = getc(ifile);
    while(! feof(ifile)){
        fseek(ifile,-1,SEEK_CUR);
        append(priv_readFromFileP(ifile,filename));
        ch = getc(ifile);
    }

    fclose(ifile);
    if(buf){
        delete buf;
    }
}

trainData trainData::priv_readFromFileP(FILE *& ifile, const std::string& filename)const{
    //include file version check
    trainData out;
    out.checkFile(ifile, filename);
    out.readNested(out.feature_shapes_, ifile);
    out.readNested(out.truth_shapes_, ifile);
    out.readNested(out.weight_shapes_, ifile);

    out.feature_arrays_ .readFromFile(ifile);
    out.truth_arrays_.readFromFile(ifile);
    out.weight_arrays_.readFromFile(ifile);
    return out;
}

void trainData::priv_readSelfFromFileP(FILE *& ifile, const std::string& filename){
    checkFile(ifile, filename);
    readNested(feature_shapes_, ifile);
    readNested(truth_shapes_, ifile);
    readNested(weight_shapes_, ifile);

    feature_arrays_ .readFromFile(ifile);
    truth_arrays_.readFromFile(ifile);
    weight_arrays_.readFromFile(ifile);
}

void trainData::readMetaDataFromFile(const std::string& filename){

    FILE *ifile = fopen(filename.data(), "rb");
    checkFile(ifile,filename);

    readNested(feature_shapes_, ifile);
    readNested(truth_shapes_, ifile);
    readNested(weight_shapes_, ifile);

    //read dtypes

    feature_arrays_ .readMetaDataFromFile(ifile);
    truth_arrays_.readMetaDataFromFile(ifile);
    weight_arrays_.readMetaDataFromFile(ifile);

    fclose(ifile);

}

std::vector<int64_t> trainData::getFirstRowsplits()const{

    const std::vector<const typeContainer* > vv = {&feature_arrays_, &truth_arrays_, &weight_arrays_};
    for(const auto& a: vv)
        for (size_t i=0;i<a->size();i++)
            if(a->at(i).rowsplits().size())
                return a->at(i).rowsplits();

    return std::vector<int64_t>();
}

std::vector<int64_t> trainData::readShapesAndRowSplitsFromFile(const std::string& filename, bool checkConsistency){
    std::vector<int64_t> rowsplits;

    FILE *ifile = fopen(filename.data(), "rb");
    checkFile(ifile,filename);

    //shapes
    std::vector<std::vector<int> > dummy;
    readNested(feature_shapes_, ifile);
    readNested(truth_shapes_, ifile);
    readNested(weight_shapes_, ifile);

    //features
    readRowSplitArray(ifile,rowsplits,checkConsistency);
    if(!checkConsistency && rowsplits.size()){
        fclose(ifile);
        return rowsplits;
    }
    //truth
    readRowSplitArray(ifile,rowsplits,checkConsistency);
    if(!checkConsistency && rowsplits.size()){
        fclose(ifile);
        return rowsplits;
    }
    //weights
    readRowSplitArray(ifile,rowsplits,checkConsistency);

    fclose(ifile);
    return rowsplits;

}

void trainData::clear() {
    feature_arrays_.clear();
    truth_arrays_.clear();
    weight_arrays_.clear();
    updateShapes();
}

void trainData::checkFile(FILE *& ifile, const std::string& filename)const{
    if(!ifile)
        throw std::runtime_error("trainData::readFromFile: file "+filename+" could not be opened.");
    float version = 0;
    io::readFromFile(&version, ifile);
    if(!checkVersionCompatible(version))
        throw std::runtime_error("trainData::readFromFile: wrong format version");

}

void trainData::readRowSplitArray(FILE *& ifile, std::vector<int64_t> &rowsplits, bool check)const{
    size_t size = 0;
    io::readFromFile(&size, ifile);
    for(size_t i=0;i<size;i++){
        auto frs = simpleArrayBase::readRowSplitsFromFileP(ifile, true);
        if(frs.size()){
            if(check){
                if(rowsplits.size() && rowsplits != frs)
                    throw std::runtime_error("trainData::readShapesAndRowSplitsFromFile: row splits inconsistent");
            }
            rowsplits=frs;
        }
    }
}

std::vector<std::vector<int> > trainData::getShapes(const typeContainer& a)const{
    std::vector<std::vector<int> > out;
    for(size_t i=0;i<a.size();i++)
        out.push_back(a.at(i).shape());
    return out;
}

void trainData::updateShapes(){

    feature_shapes_ = getShapes(feature_arrays_);
    truth_shapes_ = getShapes(truth_arrays_);
    weight_shapes_ = getShapes(weight_arrays_);

}

void trainData::skim(size_t batchelement){
    if(batchelement > nElements())
        throw std::out_of_range("trainData<T>::skim: batch element out of range");
    *this = getSlice(batchelement,batchelement+1);
}


boost::python::list trainData::transferNamesToPyList(const typeContainer& tc)const{
    boost::python::list out;
    for(size_t i=0;i<tc.size();i++){
        auto name = tc.at(i).name();
        if(! name.length()){
            name = std::to_string(i);//set a default name
        }
        out.append(name);
        if(tc.at(i).isRagged())
            out.append(name+"_rowsplits");
    }
    return out;
}

boost::python::list trainData::transferShapesToPyList(const std::vector<std::vector<int> >& vs)const{
    boost::python::list out;
    for(const auto& a: vs){
        boost::python::list nlist;
        bool wasragged=false;
        for(size_t i=1;i<a.size();i++){
            if(a.at(i)<0){
                nlist = boost::python::list();//ignore everything before
                wasragged=true;
            }
            else
                nlist.append(std::abs(a.at(i)));
        }
        out.append(nlist);
        if(wasragged){
            boost::python::list rslist;
            rslist.append(1);
            out.append(rslist);
        }
    }
    return out;
}

boost::python::list trainData::transferDTypesToPyList(const typeContainer& tc)const{
    boost::python::list out;
    for(size_t k=0;k<tc.size();k++){
        const auto& a = tc.at(k).shape();

        bool isragged=false;
        for(size_t i=0;i<a.size();i++){
            if(a.at(i)<0){
                isragged=true;
                break;
            }
        }
        out.append(tc.at(k).dtypeString());
        if(isragged)
            out.append("int64");
    }
    return out;
}


boost::python::list trainData::getTruthRaggedFlags()const{
    boost::python::list out;
    for(const auto& a: truth_shapes_){
        bool isragged = false;
        for(const auto & s: a)
            if(s<0){
                isragged=true;
                break;
            }
        if(isragged)
            out.append(true);
        else
            out.append(false);
    }
    return out;
}

boost::python::list trainData::transferToNumpyList(typeContainer& c, bool pad_rowsplits){
    namespace p = boost::python;
    namespace np = boost::python::numpy;
    p::list out;
    for(size_t i=0;i<c.size();i++){
        auto& a = c.at(i);
        if(a.isRagged()){
            auto arrt = a.transferToNumpy(pad_rowsplits);//pad row splits
            out.append(arrt[0]);//data
            np::ndarray rs = boost::python::extract<np::ndarray>(arrt[1]);
            out.append(rs.reshape(p::make_tuple(-1,1)));//row splits
        }
        else
            out.append(a.transferToNumpy(false)[0]);
    }
    return out;
}


boost::python::list trainData::transferFeatureListToNumpy(bool padrowsplits){
    return transferToNumpyList(feature_arrays_,padrowsplits);
}

boost::python::list trainData::transferTruthListToNumpy(bool padrowsplits){
    return transferToNumpyList(truth_arrays_,padrowsplits);
}

boost::python::list trainData::transferWeightListToNumpy(bool padrowsplits){
    return transferToNumpyList(weight_arrays_,padrowsplits);
}

}//ns
