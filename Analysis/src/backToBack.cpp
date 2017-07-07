#define backToBack_cxx
#include "backToBack.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <iostream>

using namespace std;

float backToBack::Dphi(float phi1, float phi2)
{
    float deltaPhi = TMath::Abs(phi1-phi2);
    if(deltaPhi > TMath::Pi())
        deltaPhi = TMath::TwoPi() - deltaPhi;
    return deltaPhi;
}

float backToBack::isBalanced(vector<float> genPt, int GenPt)
{
    // for single jet events return 0, i.e. do not use them
    if (genPt.size() < 2) return 0;
    // sort by PT to have PT descendin vector
    sort(genPt.begin(),genPt.end(),[](float a, float b) {return a > b; });
    // check if first two jets are very differnt in PT (factor 2)
    if ( genPt[0]> 2*genPt[1]) {

        //cout << genPt[0]<< " "<< genPt[1]<< " 2 jets unbalenced" <<endl;
        return 0;
    }
    // if only dijet are there return
    if (genPt.size() == 2)  {
        //cout << genPt[0]<< " "<< genPt[1]<< " 2jets  balenced" <<endl;
        return 1;
    }
    // for >=3 jets, check that the 3rd jet only had 15% of the first too (Yuta recipe)
    if(genPt.size() >= 3){
        //cout << genPt[0]<< " "<< genPt[1]<< " "<< genPt[2]<<  " so " << 1./genPt[2]*(genPt[0]+genPt[1]) << endl;
        if(genPt[2] < 0.15*(genPt[0]+genPt[1])) return 1.;
    }
    return 0.;
}



void backToBack::Loop()
{
    //   In a ROOT session, you can do:
    //      root> .L backToBack.C
    //      root> backToBack t
    //      root> t.GetEntry(12); // Fill t data members with entry number 12
    //      root> t.Show();       // Show values of entry 12
    //      root> t.Show(16);     // Read and show values of entry 16
    //      root> t.Loop();       // Loop on all entries
    //

    //     This is the loop skeleton where:
    //    jentry is the global entry number in the chain
    //    ientry is the entry number in the current Tree
    //  Note that the argument to GetEntry must be:
    //    jentry for TChain::GetEntry
    //    ientry for TTree::GetEntry and TBranch::GetEntry
    //
    //       To read only selected branches, Insert statements like:
    // METHOD1:
    //    fChain->SetBranchStatus("*",0);  // disable all branches
    //    fChain->SetBranchStatus("branchname",1);  // activate branchname
    // METHOD2: replace line
    //    fChain->GetEntry(jentry);       //read all branches
    //by  b_branchname->GetEntry(ientry); //read only this branch
    if (fChain == 0) return;

    Long64_t nentries = fChain->GetEntriesFast();

    // The tree to stare the flag if the jet is to be removed
    TFile hfile(outfile,"RECREATE");
    TTree *killtree = new TTree("tree","tree");


    vector<float>  genPT;
    float keep;
    float jetno =0 ;
    unsigned int lastEvent=0;
    // booking the output tree
    killtree->Branch("keep",&keep,"keep/F");
    Long64_t nbytes = 0, nb = 0;
    //cout << "WARNING: for testing reading just a few events"<<endl;
    //nentries = 100;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        //cout << jentry <<endl;
        nb = fChain->GetEntry(jentry);   nbytes += nb;
        //cout << "event: "<< jentry << " "<<event_no<<endl;
        // check for new event

        if(lastEvent!=event_no)
        {
            //cout << " a new event! Last even jet no:"<< jetno<< endl;
            lastEvent = event_no;
            //if (jetno!=genPT.size()) //cout << "cannot count:"<< jetno <<" "<< genPT.size() <<endl;
            if(jetno==1)
            {
                // fill a 0 is only a single jet
                keep=0;
                //cout << " filling "<< keep<<endl;
                killtree->Fill();
            }
            else{
                // else loop over kets and fill "balance" test output
                keep = isBalanced(genPT,jetno);
                for (int i =0 ; i< jetno; i++)
                {
                    //cout << " filling "<< keep<<endl;
                    killtree->Fill();
                }
            }
            genPT.clear();
            jetno=0;
            // after having Filled the tree, start new list of gen PT jets
            genPT.push_back(gen_pt);
            jetno++;
            //cout << "this is jet: "<< jetno<<endl;
            if ( killtree->GetEntries()!=jentry)
                cout << "OUT of sync" << killtree->GetEntries() <<" " << jentry<< endl;
        }

        else
        {
            genPT.push_back(gen_pt);
            jetno++;
            //cout << "this is jet: "<< jetno<<endl;
        }


        if (jentry==nentries-1)
        {
            keep=0;
            cout << " the last event "<< jetno << " "   << endl;
            for (int i =0 ; i< jetno; i++)
            {
                cout << " filling "<< i <<" "<< keep<<endl;
                killtree->Fill();
            }
            if ( killtree->GetEntries()!=jentry+1) cout << "OUT of sync" << killtree->GetEntries() <<" " << jentry+1<< endl;
        }

    }

    killtree->Write();
    hfile.Close();
    // if (Cut(ientry) < 0) continue;
}

