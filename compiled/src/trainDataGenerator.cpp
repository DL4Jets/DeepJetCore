#include "../interface/trainDataGenerator.h"

namespace djc{


trainDataGenerator::trainDataGenerator() :debuglevel(0),
        randomcount_(1), batchsize_(2),sqelementslimit_(false),skiplargebatches_(true), readthread_(0), nextreadIdx_(0), filecount_(0), nbatches_(
                0), npossiblebatches_(0), ntotal_(0), nsamplesprocessed_(0),lastbatchsize_(0),filetimeout_(10),
                batchcount_(0),lastbuffersplit_(0){
}

trainDataGenerator::~trainDataGenerator(){
    if(readthread_){
        readthread_->join();
        delete readthread_;
    }

}

void trainDataGenerator::setFileListPy(boost::python::list files){
    trainDataGenerator::setFileList(toSTLVector<std::string>(files));
}

void trainDataGenerator::shuffleFileList(){
    std::random_device rd;
    std::mt19937 g(rd());
    g.seed(randomcount_);
    randomcount_++;
    std::shuffle(std::begin(shuffle_indices_),std::end(shuffle_indices_),g);

    for(const auto i:shuffle_indices_)
        std::shuffle(std::begin(sub_shuffle_indices_.at(i)),
                std::end(sub_shuffle_indices_.at(i)),g);

    //redo splits etc
    prepareSplitting();
    batchcount_=0;
    lastbuffersplit_=0;
}

void trainDataGenerator::setBuffer(const trainData& td){

    clear();
    if(td.featureShapes().size()<1 || td.featureShapes().at(0).size()<1)
        throw std::runtime_error("trainDataGenerator<T>::setBuffer: no features filled in trainData object");
    auto hasRagged = tdHasRaggedDimension(td);

    auto rs = td.getFirstRowsplits();
    if(rs.size())
        orig_rowsplits_.push_back(rs);
    shuffle_indices_.push_back(0);
    std::vector<size_t> vec;
    for(size_t i=0;i<td.nElements();i++)
        vec.push_back(i);
    sub_shuffle_indices_.push_back(vec);
    ntotal_ = td.nElements();
    buffer_store=td;
    lastbuffersplit_=0;
    prepareSplitting();
}

void trainDataGenerator::readBuffer(){ //inject by file shuffle here
    size_t ntries = 0;
    std::exception caught;
    while(ntries < filetimeout_){
        if(io::fileExists(nextread_)){
            try{
                if(debuglevel>0)
                    std::cout << "reading file " << nextread_ << std::endl;
                //use mem buffered read, read whole file in one go and then decompress etc from memory
                buffer_read.readFromFileBuffered(nextread_);
                if(debuglevel>0)
                    std::cout << "reading file " << nextread_ << " done"<< std::endl;
                buffer_read = buffer_read.shuffle(sub_shuffle_indices_.at(nextreadIdx_));
                return;
            }
            catch(std::exception & e){ //if there are data glitches we don't want the whole training fail immediately
                caught=e;
                std::cout << "File not "<< nextread_ <<" successfully read: " << e.what() << std::endl;
                std::cout << "trying " << filetimeout_-ntries << " more time(s)" << std::endl;
                ntries+=1;
            }
        }
        sleep(1);
        ntries++;
    }
    buffer_read.clear();
    throw std::runtime_error("trainDataGenerator<T>::readBuffer: file "+nextread_+ " could not be read.");
}

void trainDataGenerator::readInfo(){
    ntotal_=0;
    bool hasRagged=false;
    bool firstfile=true;

    shuffle_indices_.resize(orig_infiles_.size());
    for(size_t i=0;i<shuffle_indices_.size();i++)
        shuffle_indices_[i]=i;

    for(const auto& f: orig_infiles_){
        trainData td;

        td.readMetaDataFromFile(f);
        //first dimension is always Nelements. At least features are filled
        if(td.featureShapes().size()<1 || td.featureShapes().at(0).size()<1)
            throw std::runtime_error("trainDataGenerator<T>::readNTotal: no features filled in trainData object "+f);

        //create sub_shuffle_idxs
        std::vector<size_t> vec;
        for(size_t i=0;i<td.nElements();i++){
            vec.push_back(i);
        }
        sub_shuffle_indices_.push_back(vec);

        if(firstfile){
            hasRagged = tdHasRaggedDimension(td);
        }
        if(hasRagged){
            std::vector<int64_t> rowsplits = td.readShapesAndRowSplitsFromFile(f, firstfile);//check consistency only for first
            if(debuglevel>1)
                std::cout << "rowsplits.size() " <<rowsplits.size() << ": "<<f <<  std::endl; //debuglevel
            orig_rowsplits_.push_back(rowsplits);
        }
        firstfile=false;
        ntotal_ += td.nElements();
    }
    if(debuglevel>0)
        std::cout << "trainDataGenerator<T>::readInfo: total elements "<< ntotal_ <<std::endl;
    batchcount_=0;
    lastbuffersplit_=0;
    prepareSplitting();
}

std::vector<int64_t> trainDataGenerator::subShuffleRowSplits(const std::vector<int64_t>& thisrs,
        const std::vector<size_t>& s_idx)const{

    auto nelems = simpleArrayBase::dataSplitToSplitIndices(thisrs);
    auto snelems=nelems;
    //shuffle
    for(size_t si=0;si<s_idx.size();si++){
        snelems.at(si) = nelems.at(s_idx.at(si));
    }
    return simpleArrayBase::splitToDataSplitIndices(snelems);

}

void trainDataGenerator::prepareSplitting(){
    splits_.clear();
    nbatches_=0;
    if(orig_rowsplits_.size()<1){//no row splits, just equal batch size except for last batch
        size_t used_events=0;
        while(used_events<ntotal_){
            if(used_events + batchsize_ <= ntotal_){
                splits_.push_back(batchsize_);
                used_events+=batchsize_;
                nbatches_++;
            }
            else{
                splits_.push_back(ntotal_-used_events);
                nbatches_++;
                break;
            }
        }
        if(debuglevel>1){
            std::cout << "trainDataGenerator<T>::prepareSplitting: splits" <<std::endl;
            for(const auto& s: splits_)
                std::cout << s << ", ";
            std::cout << std::endl;
        }
        return;
    }

    ///////row splits part

    std::vector<int64_t> allrs;
    for(size_t i=0;i<orig_rowsplits_.size();i++){
        auto shuffled_idx = shuffle_indices_.at(i);
        auto thisrs = orig_rowsplits_.at(shuffled_idx); //inject by file shuffle here
        thisrs = subShuffleRowSplits(thisrs, sub_shuffle_indices_.at(shuffled_idx));

        if(i==0 || allrs.size()==0){
            allrs=thisrs;}
        else{
            allrs = simpleArrayBase::mergeRowSplits(allrs,thisrs);
        }
    }

    if(debuglevel>1){
        std::cout << "all (first 100) row splits " <<  allrs.size() << std::endl;
        int counter =0;
        for(const auto& s: allrs){
            std::cout << s << ", " ;
            if(counter>100)break;
            counter++;
        }
        std::cout << std::endl;
    }
    std::vector<size_t> nelems_per_split;
    splits_ = simpleArrayBase::getSplitIndices(allrs, batchsize_,sqelementslimit_ , skiplargebatches_, usebatch_, nelems_per_split);

    nbatches_=0;
    npossiblebatches_=0;
    for(size_t i=0;i<usebatch_.size();i++){
        npossiblebatches_++;
        if(usebatch_.at(i))
            nbatches_++;
    }


    if(debuglevel>1){
        size_t nprint = splits_.size();
        if(nprint>200)nprint=200;
        for(size_t i=0;i< nprint;i++){
            std::cout << i ;
            if(usebatch_.at(i))
                std::cout << " ok, split " ;
            else
                std::cout << " no, split ";
            std::cout << splits_.at(i) << "; nelements "<< nelems_per_split.at(i)<< std::endl;
        }
        std::cout << std::endl;
    }

}

bool trainDataGenerator::tdHasRaggedDimension(const trainData& td)const{
    for(const auto& sv: td.featureShapes())
        for(const auto& s:sv)
            if(s<0)
                return true;
    for(const auto& sv: td.truthShapes())
        for(const auto& s:sv)
            if(s<0)
                return true;
    for(const auto& sv: td.weightShapes())
        for(const auto& s:sv)
            if(s<0)
                return true;
    return false;
}

bool trainDataGenerator::lastBatch()const{
    return batchcount_ >= npossiblebatches_ -1 ;
}

bool trainDataGenerator::isEmpty()const{
    return batchcount_ >= splits_.size();
}

void trainDataGenerator::prepareNextEpoch(){

    //prepare for next epoch, pre-read first file
    if(readthread_){
        readthread_->join(); //this is slow! FIXME: better way to exit gracefully in a simple way
        delete readthread_;

    }
    buffer_store.clear();
    buffer_read.clear();
    filecount_=0;
    nsamplesprocessed_=0;
    batchcount_=0;
    lastbatchsize_=0;
    lastbuffersplit_=0;
    nextreadIdx_ = shuffle_indices_.at(filecount_);
    nextread_ = orig_infiles_.at(nextreadIdx_);
    filecount_++;
    readthread_ = new std::thread(&trainDataGenerator::readBuffer,this);

}

void trainDataGenerator::end(){
    if(readthread_){
        readthread_->join(); //this is slow! FIXME: better way to exit gracefully in a simple way
        delete readthread_;
        readthread_=0;
    }
}

void trainDataGenerator::clear(){
    end();
    orig_infiles_.clear();
    shuffle_indices_.clear();
    sub_shuffle_indices_.clear();
    orig_rowsplits_.clear();
    splits_.clear();
    usebatch_.clear();
    randomcount_=0;

    //batchsize_ keep batch size
    //sqelementslimit_ keep
    //skiplargebatches_ keep
    buffer_store.clear();
    buffer_read.clear();

    filecount_=0;
    nbatches_=0;
    ntotal_=0;
    nsamplesprocessed_=0;
    lastbatchsize_=0;
    lastbuffersplit_=0;
    // filetimeout_ keep
    batchcount_=0;
}

trainData trainDataGenerator::getBatch(){
    return prepareBatch();
}

trainData  trainDataGenerator::prepareBatch(){
    if(isEmpty()){
        std::cout << "trainDataGenerator::prepareBatch: batchcount " << batchcount_ << ", available: " << splits_.size() << std::endl;
        throw std::runtime_error("trainDataGenerator::prepareBatch: asking for more batches than in dataset");
    }

    size_t bufferelements=buffer_store.nElements();
    size_t expect_batchelements = splits_.at(batchcount_);
    bool usebatch = true;

    if(!expect_batchelements)//sanity check
        throw std::runtime_error("trainDataGenerator<T>::prepareBatch: expected elements zero!");

    if(usebatch_.size())
        usebatch = usebatch_.at(batchcount_);

    if(debuglevel>2)
        std::cout << "expect_batchelements "<<expect_batchelements << " vs " << bufferelements-lastbuffersplit_ <<" bufferelements" << std::endl;

    while(bufferelements-lastbuffersplit_<expect_batchelements){
        //if thread, read join
        if(readthread_){
            readthread_->join();
            delete readthread_;
            readthread_=0;
        }
        if(lastbuffersplit_)
            if(lastbuffersplit_ != buffer_store.nElements()){
                buffer_store = buffer_store.getSlice(lastbuffersplit_,buffer_store.nElements());//cut the front part
                buffer_store.append(buffer_read);
            }
            else{ //was used completely
                buffer_store = buffer_read;//std::move(buffer_read); //possible opt. implement move for trainData fully
            }
        else{ //first one
            buffer_store.append(buffer_read);//std::move(buffer_read);
        }
        buffer_read.clear();
        bufferelements = buffer_store.nElements();
        lastbuffersplit_=0;

        if(debuglevel>2)
            std::cout << "nprocessed " << nsamplesprocessed_ << " file " << filecount_ << " in buffer " << bufferelements
            << " file read " << nextread_ << " totalfiles " << orig_infiles_.size()
            << " total events "<< ntotal_<< std::endl;

        if(nsamplesprocessed_ + bufferelements < ntotal_){
            if (filecount_ >= orig_infiles_.size()){
                std::cout << "trainDataGenerator<T>::prepareBatch: filecount: "<<  filecount_ <<" infiles "<< orig_infiles_.size()<<
                        " processed: "<< nsamplesprocessed_ << " buffer:  "<< bufferelements << " total "<< ntotal_ << std::endl;
                throw std::runtime_error(
                        "trainDataGenerator<T>::prepareBatch: more file reads requested than batches in the sample");

            }

            nextreadIdx_ = shuffle_indices_.at(filecount_);
            nextread_ = orig_infiles_.at(nextreadIdx_);

            if(debuglevel>0)
                std::cout << "start new read on file "<< nextread_ <<std::endl;

            filecount_++;
            readthread_ = new std::thread(&trainDataGenerator::readBuffer,this);
        }
    }

    if( ! buffer_store.validSlice(lastbuffersplit_, lastbuffersplit_+expect_batchelements)){
        throw std::runtime_error("trainDataGenerator::prepareBatch: split error");
    }

    //auto thisbatch = buffer_store.split(expect_batchelements);
    auto thisbatch = buffer_store.getSlice(lastbuffersplit_, lastbuffersplit_+expect_batchelements);

    lastbuffersplit_+=expect_batchelements;
    // validSlice

    if(thisbatch.nTotalElements() < 1){
      //not sure why this can happen, there might be some bigger problem here. This at least prevents crashes.
   //   return prepareBatch();
    }

    if(debuglevel>2)
        std::cout << "providing batch " << nsamplesprocessed_ << "-" << nsamplesprocessed_+expect_batchelements <<
        ", slice " << lastbuffersplit_-expect_batchelements << "-" << lastbuffersplit_ <<
        "\nelements in buffer before: " << bufferelements <<
        "\nsplitting at " << expect_batchelements << " use this batch "<<  usebatch
        << " total elements " << thisbatch.nTotalElements() << " elements left in buffer " << buffer_store.nElements()<< std::endl;

    if(debuglevel>3){
        int dbpcount=0;
        for(const auto& s: buffer_store.featureArray(0).rowsplits()){
            std::cout << s << ", ";
            if(dbpcount>50)break;
            dbpcount++;
        }
        std::cout << std::endl;
    }

    nsamplesprocessed_+=expect_batchelements;
    lastbatchsize_ = expect_batchelements;

    batchcount_++;
    if(! usebatch){//until valid batch
        return prepareBatch();
    }

    return thisbatch;

}

}//ns
