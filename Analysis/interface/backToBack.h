//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Jul  3 16:42:25 2017 by ROOT version 6.06/01
// from TTree tree/tree
// found on file: ntuple_qcd_30_50_phase1_98.root
//////////////////////////////////////////////////////////

#ifndef backToBack_h
#define backToBack_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <vector>

//////////root requires it.... not good style!!
using namespace std;
// Header file for the classes stored in the TTree if any.

class backToBack {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.
   const Int_t kMaxsv_num = 4;
   const Int_t kMaxnsv = 1;
   const Int_t kMaxsv_pt = 1;
   const Int_t kMaxsv_eta = 1;
   const Int_t kMaxsv_phi = 1;
   const Int_t kMaxsv_etarel = 1;
   const Int_t kMaxsv_phirel = 1;
   const Int_t kMaxsv_deltaR = 1;
   const Int_t kMaxsv_mass = 1;
   const Int_t kMaxsv_ntracks = 1;
   const Int_t kMaxsv_chi2 = 1;
   const Int_t kMaxsv_ndf = 1;
   const Int_t kMaxsv_normchi2 = 1;
   const Int_t kMaxsv_dxy = 1;
   const Int_t kMaxsv_dxyerr = 1;
   const Int_t kMaxsv_dxysig = 1;
   const Int_t kMaxsv_d3d = 1;
   const Int_t kMaxsv_d3err = 1;
   const Int_t kMaxsv_d3dsig = 1;
   const Int_t kMaxsv_costhetasvpv = 1;
   const Int_t kMaxsv_enratio = 1;
   const Int_t kMaxLooseIVF_sv_num = 4;
   const Int_t kMaxLooseIVF_nsv = 1;
   const Int_t kMaxLooseIVF_sv_pt = 1;
   const Int_t kMaxLooseIVF_sv_eta = 1;
   const Int_t kMaxLooseIVF_sv_phi = 1;
   const Int_t kMaxLooseIVF_sv_etarel = 1;
   const Int_t kMaxLooseIVF_sv_phirel = 1;
   const Int_t kMaxLooseIVF_sv_deltaR = 1;
   const Int_t kMaxLooseIVF_sv_mass = 1;
   const Int_t kMaxLooseIVF_sv_ntracks = 1;
   const Int_t kMaxLooseIVF_sv_chi2 = 1;
   const Int_t kMaxLooseIVF_sv_ndf = 1;
   const Int_t kMaxLooseIVF_sv_normchi2 = 1;
   const Int_t kMaxLooseIVF_sv_dxy = 1;
   const Int_t kMaxLooseIVF_sv_dxyerr = 1;
   const Int_t kMaxLooseIVF_sv_dxysig = 1;
   const Int_t kMaxLooseIVF_sv_d3d = 1;
   const Int_t kMaxLooseIVF_sv_d3err = 1;
   const Int_t kMaxLooseIVF_sv_d3dsig = 1;
   const Int_t kMaxLooseIVF_sv_costhetasvpv = 1;
   const Int_t kMaxLooseIVF_sv_enratio = 1;
   const Int_t kMaxgen_pt = 1;
   const Int_t kMaxDelta_gen_pt = 1;
   const Int_t kMaxisB = 1;
   const Int_t kMaxisBB = 1;
   const Int_t kMaxisLeptonicB = 1;
   const Int_t kMaxisLeptonicB_C = 1;
   const Int_t kMaxisC = 1;
   const Int_t kMaxisUD = 1;
   const Int_t kMaxisS = 1;
   const Int_t kMaxisG = 1;
   const Int_t kMaxisUndefined = 1;
   const Int_t kMaxisPhysB = 1;
   const Int_t kMaxisPhysBB = 1;
   const Int_t kMaxisPhysLeptonicB = 1;
   const Int_t kMaxisPhysLeptonicB_C = 1;
   const Int_t kMaxisPhysC = 1;
   const Int_t kMaxisPhysUD = 1;
   const Int_t kMaxisPhysS = 1;
   const Int_t kMaxisPhysG = 1;
   const Int_t kMaxisPhysUndefined = 1;
   const Int_t kMaxmuons_number = 6;
   const Int_t kMaxelectrons_number = 4;
   const Int_t kMaxmuons_isLooseMuon = 1;
   const Int_t kMaxmuons_isTightMuon = 1;
   const Int_t kMaxmuons_isSoftMuon = 1;
   const Int_t kMaxmuons_isHighPtMuon = 1;
   const Int_t kMaxmuons_pt = 1;
   const Int_t kMaxmuons_relEta = 1;
   const Int_t kMaxmuons_relPhi = 1;
   const Int_t kMaxmuons_energy = 1;
   const Int_t kMaxelectrons_pt = 1;
   const Int_t kMaxelectrons_relEta = 1;
   const Int_t kMaxelectrons_relPhi = 1;
   const Int_t kMaxelectrons_energy = 1;
   const Int_t kMaxgen_pt_Recluster = 1;
   const Int_t kMaxgen_pt_WithNu = 1;
   const Int_t kMaxDelta_gen_pt_Recluster = 1;
   const Int_t kMaxDelta_gen_pt_WithNu = 1;
   const Int_t kMaxn_Cpfcand = 30;
   const Int_t kMaxnCpfcand = 1;
   const Int_t kMaxCpfcan_pt = 1;
   const Int_t kMaxCpfcan_eta = 1;
   const Int_t kMaxCpfcan_phi = 1;
   const Int_t kMaxCpfcan_ptrel = 1;
   const Int_t kMaxCpfcan_erel = 1;
   const Int_t kMaxCpfcan_phirel = 1;
   const Int_t kMaxCpfcan_etarel = 1;
   const Int_t kMaxCpfcan_deltaR = 1;
   const Int_t kMaxCpfcan_puppiw = 1;
   const Int_t kMaxCpfcan_dxy = 1;
   const Int_t kMaxCpfcan_dxyerrinv = 1;
   const Int_t kMaxCpfcan_dxysig = 1;
   const Int_t kMaxCpfcan_dz = 1;
   const Int_t kMaxCpfcan_VTX_ass = 1;
   const Int_t kMaxCpfcan_fromPV = 1;
   const Int_t kMaxCpfcan_drminsv = 1;
   const Int_t kMaxCpfcan_vertex_rho = 1;
   const Int_t kMaxCpfcan_vertex_phirel = 1;
   const Int_t kMaxCpfcan_vertex_etarel = 1;
   const Int_t kMaxCpfcan_dptdpt = 1;
   const Int_t kMaxCpfcan_detadeta = 1;
   const Int_t kMaxCpfcan_dphidphi = 1;
   const Int_t kMaxCpfcan_dxydxy = 1;
   const Int_t kMaxCpfcan_dzdz = 1;
   const Int_t kMaxCpfcan_dxydz = 1;
   const Int_t kMaxCpfcan_dphidxy = 1;
   const Int_t kMaxCpfcan_dlambdadz = 1;
   const Int_t kMaxCpfcan_BtagPf_trackMomentum = 1;
   const Int_t kMaxCpfcan_BtagPf_trackEta = 1;
   const Int_t kMaxCpfcan_BtagPf_trackEtaRel = 1;
   const Int_t kMaxCpfcan_BtagPf_trackPtRel = 1;
   const Int_t kMaxCpfcan_BtagPf_trackPPar = 1;
   const Int_t kMaxCpfcan_BtagPf_trackDeltaR = 1;
   const Int_t kMaxCpfcan_BtagPf_trackPtRatio = 1;
   const Int_t kMaxCpfcan_BtagPf_trackSip3dVal = 1;
   const Int_t kMaxCpfcan_BtagPf_trackSip3dSig = 1;
   const Int_t kMaxCpfcan_BtagPf_trackSip2dVal = 1;
   const Int_t kMaxCpfcan_BtagPf_trackSip2dSig = 1;
   const Int_t kMaxCpfcan_BtagPf_trackDecayLen = 1;
   const Int_t kMaxCpfcan_BtagPf_trackJetDistVal = 1;
   const Int_t kMaxCpfcan_BtagPf_trackJetDistSig = 1;
   const Int_t kMaxCpfcan_isMu = 1;
   const Int_t kMaxCpfcan_isEl = 1;
   const Int_t kMaxCpfcan_chi2 = 1;
   const Int_t kMaxCpfcan_quality = 1;
   const Int_t kMaxn_Npfcand = 37;
   const Int_t kMaxNpfcan_pt = 1;
   const Int_t kMaxNpfcan_eta = 1;
   const Int_t kMaxNpfcan_phi = 1;
   const Int_t kMaxNpfcan_ptrel = 1;
   const Int_t kMaxNpfcan_erel = 1;
   const Int_t kMaxNpfcan_puppiw = 1;
   const Int_t kMaxNpfcan_phirel = 1;
   const Int_t kMaxNpfcan_etarel = 1;
   const Int_t kMaxNpfcan_deltaR = 1;
   const Int_t kMaxNpfcan_isGamma = 1;
   const Int_t kMaxNpfcan_HadFrac = 1;
   const Int_t kMaxNpfcan_drminsv = 1;
   const Int_t kMaxtrackJetPt = 1;
   const Int_t kMaxjetNTracks = 1;
   const Int_t kMaxjetNSecondaryVertices = 1;
   const Int_t kMaxtrackSumJetEtRatio = 1;
   const Int_t kMaxtrackSumJetDeltaR = 1;
   const Int_t kMaxvertexCategory = 1;
   const Int_t kMaxtrackSip2dValAboveCharm = 1;
   const Int_t kMaxtrackSip2dSigAboveCharm = 1;
   const Int_t kMaxtrackSip3dValAboveCharm = 1;
   const Int_t kMaxtrackSip3dSigAboveCharm = 1;
   const Int_t kMaxn_jetNSelectedTracks = 12;
   const Int_t kMaxjetNSelectedTracks = 1;
   const Int_t kMaxtrackPtRel = 1;
   const Int_t kMaxtrackDeltaR = 1;
   const Int_t kMaxtrackPtRatio = 1;
   const Int_t kMaxtrackSip3dSig = 1;
   const Int_t kMaxtrackSip2dSig = 1;
   const Int_t kMaxtrackDecayLenVal = 1;
   const Int_t kMaxtrackJetDistVal = 1;
   const Int_t kMaxn_jetNTracksEtaRel = 15;
   const Int_t kMaxjetNTracksEtaRel = 1;
   const Int_t kMaxtrackEtaRel = 1;
   const Int_t kMaxtrackPParRatio = 1;
   const Int_t kMaxtrackSip2dVal = 1;
   const Int_t kMaxtrackSip3dVal = 1;
   const Int_t kMaxtrackMomentum = 1;
   const Int_t kMaxtrackEta = 1;
   const Int_t kMaxtrackPPar = 1;
   const Int_t kMaxn_StoredVertices = 1;
   const Int_t kMaxNStoredVertices = 1;
   const Int_t kMaxvertexMass = 1;
   const Int_t kMaxvertexNTracks = 1;
   const Int_t kMaxvertexEnergyRatio = 1;
   const Int_t kMaxvertexJetDeltaR = 1;
   const Int_t kMaxflightDistance2dVal = 1;
   const Int_t kMaxflightDistance2dSig = 1;
   const Int_t kMaxflightDistance3dVal = 1;
   const Int_t kMaxflightDistance3dSig = 1;

   // Declaration of leaf types
  
   UInt_t          event_no;
   UInt_t          jet_no;
   Float_t         gen_pt;
   Float_t         jet_pt;
   Float_t         jet_corr_pt;
   Float_t         jet_eta;
   Float_t         jet_phi;
  

   // List of branches
  
   TBranch        *b_event_no;   //!
   TBranch        *b_jet_no;   //!
   TBranch        *b_gen_pt;   //!
   TBranch        *b_jet_pt;   //!
   TBranch        *b_jet_corr_pt;   //!
   TBranch        *b_jet_eta;   //!
   TBranch        *b_jet_phi;   //!
 

   backToBack(TTree *tree=0);
   float  Dphi(float phi1, float phi2);
   virtual ~backToBack();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   float isBalanced(vector<float> genPt, int GenPt);
   TString outfile;
};

#endif

#ifdef backToBack_cxx
backToBack::backToBack(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("ntuple_qcd_80_120_phase1_99.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("ntuple_qcd_80_120_phase1_99.root");
      }
      TDirectory * dir = (TDirectory*)f->Get("ntuple_qcd_80_120_phase1_99.root:/deepntuplizer");
      dir->GetObject("tree",tree);

   }
   Init(tree);
}


backToBack::~backToBack()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t backToBack::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t backToBack::LoadTree(Long64_t entry)
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

void backToBack::Init(TTree *tree)
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

 
   fChain->SetBranchAddress("event_no", &event_no, &b_event_no);
   fChain->SetBranchAddress("jet_no", &jet_no, &b_jet_no);
   fChain->SetBranchAddress("gen_pt", &gen_pt, &b_gen_pt);
   fChain->SetBranchAddress("jet_pt", &jet_pt, &b_jet_pt);
   fChain->SetBranchAddress("jet_corr_pt", &jet_corr_pt, &b_jet_corr_pt);
   fChain->SetBranchAddress("jet_eta", &jet_eta, &b_jet_eta);
   fChain->SetBranchAddress("jet_phi", &jet_phi, &b_jet_phi);
 
   Notify();
}

Bool_t backToBack::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void backToBack::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t backToBack::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef backToBack_cxx
