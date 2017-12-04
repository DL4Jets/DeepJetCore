

//root hack-'macro' to load a full tree association for interactive plotting

//int main(){

#include "TROOT.h"
#include "../src/friendTreeInjector.cpp"
#include "TSystem.h"
friendTreeInjector * injector=0;
TChain* tree=0;

void  loadTreeAssociation(){
    TString __infile=getenv("ROOT_TREEASSOCIATIONINFILE");
    injector = new friendTreeInjector();
    injector->addFromFile(__infile);
    injector->createChain();
    tree= injector->getChain();
}
