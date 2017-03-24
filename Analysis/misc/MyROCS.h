//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Sep 23 10:40:36 2016 by ROOT version 5.34/36
// from TTree ttree/ttree
// found on file: /afs/cern.ch/work/m/mstoye/DeepBs/CMSSW_8_0_12/src/RecoBTag/TagVarExtractor/test/JetTaggingVariables.root
//////////////////////////////////////////////////////////

#ifndef MyROCS_h
#define MyROCS_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TString.h>
// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class MyROCS {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
  
   Float_t         jet_pt;
   Float_t         jet_eta;
   Float_t         jet_qgl;
   UInt_t          isB;
   UInt_t          isC;
   UInt_t          isUDS;
   UInt_t          isG;
   Float_t         Delta_gen_pt_WithNu;
   Float_t         TagVarCSV_trackSumJetEtRatio;
   Float_t         TagVarCSV_trackSumJetDeltaR;
   Float_t         TagVarCSV_vertexCategory;


   TBranch        *b_isB_;   //!
   TBranch        *b_isC_;   //!
   TBranch        *b_isUDS_;   //!
   TBranch        *b_isG_;   //!
   TBranch        *b_jet_pt;   //!
   TBranch        *b_jet_eta;   //!
   TBranch        *b_jet_qgl;   //!
   TBranch        *b_Delta_gen_pt_WithNu_; 
   TBranch        *b_trackSumJetEtRatio_;   //!
   TBranch        *b_trackSumJetDeltaR_;   //!
   TBranch        *b_vertexCategory_;   //!
   MyROCS(TTree *tree=0);
   virtual ~MyROCS();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(TString deepFlavName,TString otherName, Bool_t CSV ,TString myPlotFile, float eta, float PT);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef MyROCS_cxx
MyROCS::MyROCS(TTree *tree) : fChain(0) 
{

  /*
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
     // TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/afs/cern.ch/work/m/mstoye/root_numpy/ttbar_small_test.root");
     //  TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/afs/cern.ch/work/m/mstoye/root_numpy/QCD/QCD_test.root");
     //      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/afs/cern.ch/work/m/mstoye/root_numpy/all/all_test.root");
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/afs/cern.ch/work/m/mstoye/root_numpy/JetTaggingVariables.root");



  if (!f || !f->IsOpen()) {
    //    f = new TFile("/afs/cern.ch/work/m/mstoye/root_numpy/QCD/QCD_test.root");
    //   f = new TFile("/afs/cern.ch/work/m/mstoye/root_numpy/all/all_test.root");
     f = new TFile("/afs/cern.ch/work/m/mstoye/root_numpy/JetTaggingVariables.root");

	     //f = new TFile("/afs/cern.ch/work/m/mstoye/root_numpy/ttbar_small_test.root");
      }
      //      TDirectory * dir = (TDirectory*)f->Get("");
      f->GetObject("tagVars/ttree",tree);

   }
  */
   Init(tree);
}

MyROCS::~MyROCS()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MyROCS::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MyROCS::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void MyROCS::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("isB", &isB, &b_isB_);
   fChain->SetBranchAddress("isC", &isC, &b_isC_);
   fChain->SetBranchAddress("isUDS", &isUDS, &b_isUDS_);
   fChain->SetBranchAddress("isG", &isG, &b_isG_);
   fChain->SetBranchAddress("jet_pt", &jet_pt, &b_jet_pt);
   fChain->SetBranchAddress("jet_eta", &jet_eta, &b_jet_eta);
   fChain->SetBranchAddress("jet_qgl", &jet_qgl, &b_jet_qgl);
   fChain->SetBranchAddress("Delta_gen_pt_WithNu", &Delta_gen_pt_WithNu, &b_Delta_gen_pt_WithNu_);
   fChain->SetBranchAddress("TagVarCSV_trackSumJetEtRatio", &TagVarCSV_trackSumJetEtRatio, &b_trackSumJetEtRatio_);
   fChain->SetBranchAddress("TagVarCSV_trackSumJetDeltaR", &TagVarCSV_trackSumJetDeltaR, &b_trackSumJetDeltaR_);
   fChain->SetBranchAddress("TagVarCSV_vertexCategory", &TagVarCSV_vertexCategory, &b_vertexCategory_);

   Notify();
}

Bool_t MyROCS::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void MyROCS::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t MyROCS::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef MyROCS_cxx
