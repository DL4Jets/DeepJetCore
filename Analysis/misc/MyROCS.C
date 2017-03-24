#define MyROCS_cxx
#include "MyROCS.h"
#include <TH2.h>
#include <TStyle.h>

#include <TGraph.h>
#include <iostream>
#include <TLegend.h>


void MyROCS::Loop(TString deepFlavName,TString myPlotFiletwo,  Bool_t CSV , TString myPlotFile,  float eta , float PT )
{

	//   In a ROOT session, you can do:
	//      Root > .L MyROCS.C
	//      Root > MyROCS t
	//      Root > t.GetEntry(12); // Fill t data members with entry number 12
	//      Root > t.Show();       // Show values of entry 12
	//      Root > t.Show(16);     // Read and show values of entry 16
	//      Root > t.Loop();       // Loop on all entries
	//

	//     This is the loop skeleton where:
	//    jentry is the global entry number in the chain
	//    ientry is the entry number in the current Tree
	//    Note that the argument to GetEntry must be:
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


	// my Histos

	TFile *FileForPlots = new TFile(myPlotFile+".root","recreate");
	TH1D* CVS_b = new TH1D("CVS_b",";Prob(b+bb);normalized to unity",100,0,1);
	TH1D* CVS_c = new TH1D("CVS_c","CMS c",100,0,1);
	TH1D* CVS_udsg = new TH1D("CVS_udsg","CMS udsg",100,0,1);
	TH1D* CVS_udscg = new TH1D("CVS_","CMS udscg",100,0,1);

	TH1D* MyTag_b = new TH1D("MyTag_b",";Prob(c);normalized to unity",100,0,1);
	TH1D* MyTag_c = new TH1D("MyTag_c","CMS c",100,0,1);
	TH1D* MyTag_udsg = new TH1D("MyTag_udsg","CMS udsg",100,0,1);
	TH1D* MyTag_udscg = new TH1D("MyTag_","C udscg",100,0,1);

	TH1D* MyTag_b_test = new TH1D("MyTag_b_test",";Prob(c)/(1-Prob(u));normalized to unity",100,0,1);
	TH1D* MyTag_c_test = new TH1D("MyTag_c_test","CMS c",100,0,1);
	TH1D* MyTag_udsg_test = new TH1D("MyTag_udsg_test","CMS udsg",100,0,1);
	TH1D* MyTag_udscg_test = new TH1D("MyTag_udscg_test","CMS udscg",100,0,1);

	TH1D* vertexcat = new TH1D("vertexcat","vertexcat",4,-0.5,3.5);

	TH2D* QCDb = new TH2D("QCDb","CMS udscg",100,0,1,100,0,1);
	TH2D* QCDc = new TH2D("QCDc","CMS udscg",100,0,1,100,0,1);
	TH2D* QCDu = new TH2D("QCDu","CMS udscg",100,0,1,100,0,1);


	TFile* myTagFile = new TFile(deepFlavName,"read");
	TTree* Tagtree =  (TTree*) myTagFile->Get("tree");
	TBranch *b_b_prob;
	float  b_prob;
	Tagtree->SetBranchAddress("probB", &b_prob, &b_b_prob);

	TBranch *b_c_prob;
	float  c_prob;
	Tagtree->SetBranchAddress("probC", &c_prob, &b_c_prob);

	TBranch *b_u_prob;
	float  u_prob;
	Tagtree->SetBranchAddress("probUDSG", &u_prob, &b_u_prob);


	TFile* myTagFile2 = new TFile(myPlotFiletwo,"read");

	TTree* Tagtree2 =  (TTree*) myTagFile2->Get("tree");
	TBranch *b_b_prob2;
	Float_t  b_prob2;
	Tagtree2->SetBranchAddress("probB", &b_prob2, &b_b_prob2);
	TBranch *b_c_prob2;
	Float_t  c_prob2;
	Tagtree2->SetBranchAddress("probC", &c_prob2, &b_c_prob2);
	TBranch *b_u_prob2;
	Float_t  u_prob2;
	Tagtree2->SetBranchAddress("probUDSG", &u_prob2, &b_u_prob2);

	if (fChain == 0) return;

	Long64_t nentries = fChain->GetEntriesFast();
	std::cout << "entries " << nentries<<" "<<Tagtree->GetEntriesFast()<<" "<<Tagtree2->GetEntriesFast()<<std::endl;

	Long64_t nbytes = 0, nb = 0;

	float yes=0;
	float all=0;
	for (Long64_t jentry=0; jentry<nentries;jentry++) {
		all +=1;
		Long64_t ientry = LoadTree(jentry);
		if (ientry < 0) break;
		nb = fChain->GetEntry(jentry);   nbytes += nb;
		// Load the other tree
		Tagtree->GetEntry(jentry);
		Tagtree2->GetEntry(jentry);
		// MTagtree->GetEntry(jentry);
		// TTagtree->GetEntry(jentry);
		if(PT>=0)
			if(jet_pt<PT) continue;;
		if (PT<0)
			if(jet_pt>PT) continue;;
		if (eta>=0)
			if(fabs(jet_eta)<eta) continue;;
		if (eta<0)
			if(fabs(jet_eta)>fabs(eta)) continue;;

		// float dahed =  c_prob+cc_prob;
		float dahed = b_prob;
		float solid =  b_prob;
		//  float dotted = Jet_cMVAv2/2.+0.5 ;
		float dotted = (b_prob2);

		// if (dotted<0)dotted=0;
		//  if (dahed<0)dahed=0;
		//  if (solid<0)solid=0;
		//  if(!(solid<.2&&solid>.15&&dahed>0.51&&dahed<0.575)) continue;
		//if(TagVarCSV_trackSumJetDeltaR>0.3&&TagVarCSV_trackSumJetEtRatio==0&&TagVarCSV_vertexCategory==2) {

		//   cout << "solid  " << solid  << " dashed " << u_prob << " dotted "<<c_prob << " " << TagVarCSV_trackSumJetDeltaR << endl;
		//   if (b_prob==0 &&  u_prob==1) continue;
		//solid=0.0;

		//  //    fChain->Show(jentry);
		//   dotted=0.0;
		//  continue;
		// }
		/*
     if(TagVarCSV_vertexCategory==2&&TagVarCSV_trackSumJetEtRatio<=0) {

       //   cout << "solid  " << solid  << " dashed " << dahed << " dotted "<<dotted << " " << TagVarCSV_trackSumJetDeltaR << endl;
       //solid=0.0;
       // fChain->Show(jentry);
        dotted=0.0;
        solid=0.0;
	dahed = 0;
       // if()
       //  continue;
     }
		 */


		if(isB) {
			CVS_b->Fill(solid);
			MyTag_b->Fill(dahed);
			MyTag_b_test->Fill(dotted);
			QCDb->Fill( solid,dahed);
			//cout << "solid  " << solid  << " dashed " << dahed << " dotted "<<dotted <<endl;
			//	cout << "ome "<<b_prob2 <<endl;
			if(b_prob >c_prob && b_prob >u_prob  ) yes+=1;
			// cout << b_prob << " " <<  bb_prob << " "  <<  c_prob  <<" "<< cc_prob  << " " << u_prob << " all: " <<  b_prob +bb_prob + c_prob  + cc_prob  + u_prob<<endl;
			//	CVS_b->Fill(solid);
			//	MyTag_b->Fill(dotted);
		}
		if(isC) {
			//	CVS_c->Fill(Mb_prob);
			//	MyTag_c->Fill(b_prob);
			CVS_c->Fill(solid);
			MyTag_c->Fill(dahed);
			MyTag_c_test->Fill(dotted);
			QCDc->Fill( solid,dahed);
			//	CVS_c->Fill(solid);
			//	MyTag_c->Fill(dotted);
			if(c_prob >b_prob && c_prob >u_prob  ) yes+=1;
		}

		if(isG+isUDS) {
			//CVS_udsg->Fill(Mb_prob);
			//MyTag_udsg->Fill(b_prob);
			CVS_udsg->Fill(solid);
			MyTag_udsg->Fill(dahed);
			MyTag_udsg_test->Fill(dotted);
			QCDu->Fill( solid,dahed);
			if(u_prob > b_prob && u_prob > c_prob  ) yes+=1;
		}

		if(!isB){
			CVS_udscg->Fill(solid);
			MyTag_udscg->Fill(dahed);
			MyTag_udscg_test->Fill(dotted);
			//CVS_udscg->Fill(Mb_prob);
			//MyTag_udscg->Fill(b_prob);
		}
		// if (Cut(ientry) < 0) continue;
	}
	cout << " accurency "<< yes/all<<endl;
	FileForPlots->cd();

	TCanvas* plots2D = new TCanvas("2D");
	plots2D->Divide(3);
	plots2D->cd(1);
	QCDb->Draw("colz");
	plots2D->cd(2);
	QCDc->Draw("colz");
	plots2D->cd(3);
	QCDu->Draw("colz");

	TCanvas* plots = new TCanvas("1D");

	TLegend* oneLeg = new  TLegend (0.4,0.6,0.8,0.9);
	oneLeg->AddEntry( CVS_b,"b+bb","l");
	oneLeg->AddEntry( CVS_c,"c","l");
	oneLeg->AddEntry( CVS_udsg,"udsg","l");
	TH1D* aframe = new TH1D("aframe",";Prob(b+bb);normalized to unity",100,0,1);
	aframe->GetYaxis()->SetRangeUser(0.0005,0.5);
	aframe->Draw();
	CVS_b->DrawNormalized("same");
	CVS_c->SetLineColor(kRed);
	CVS_c->DrawNormalized("same");
	CVS_udsg->SetLineColor(kGreen);
	CVS_udsg->DrawNormalized("same");
	//   CVS_udscg->DrawNormalized("same");
	oneLeg->Draw("same");
	plots->SetLogy();
	//  plots->GetXaxis()->SetRangeUser(0.001,0.5);
	plots->Write();
	plots->Print(myPlotFile+"_DeepFL.pdf");

	TCanvas* plots2 = new TCanvas("1DMVA");
	//plots2->GetXaxis()->SetRangeUser(0.001,0.5);
	TH1D* aframe2 = new TH1D("aframe2",";Prob(c);normalized to unity",100,0,1);
	aframe2->GetYaxis()->SetRangeUser(0.0005,0.5);
	aframe2->Draw();
	MyTag_b->DrawNormalized("same");
	// MyTag_b->GetXaxis()->SetRangeUser(0.001,0.5);
	MyTag_c->SetLineColor(kRed);
	MyTag_c->DrawNormalized("same");
	MyTag_udsg->SetLineColor(kGreen);
	MyTag_udsg->DrawNormalized("same");
	// MyTag_udscg->DrawNormalized("same");
	oneLeg->Draw("same");
	plots2->SetLogy();

	plots2->Write();
	plots2->Print(myPlotFile+"_CSVIVF.pdf");

	TCanvas* plots3 = new TCanvas("other");
	TH1D* aframe3 = new TH1D("aframe3",";prob(c);normalized to unity",100,0,1);
	aframe3->GetYaxis()->SetRangeUser(0.0005,0.5);
	aframe3->Draw();

	MyTag_b_test->DrawNormalized("same");
	MyTag_b_test->GetXaxis()->SetRangeUser(0.001,0.5);
	MyTag_c_test->SetLineColor(kRed);

	MyTag_c_test->DrawNormalized("same");
	MyTag_udsg_test->SetLineColor(kGreen);
	MyTag_udsg_test->DrawNormalized("same");
	// MyTag_udscg_test->DrawNormalized("same");
	plots3->SetLogy();
	//  plots3->GetXaxis()->SetRangeUser(0.001,0.5);
	oneLeg->Draw("same");
	plots3->Write();
	plots3->Print(myPlotFile+"_aMVAv2.pdf");
	// TCanvas* plots4 = new TCanvas("vertex");
	//  vertexcat->Draw();

	Double_t x_b[100];
	Double_t x_c[100];
	Double_t x_udsg[100];
	Double_t x_udscg[100];

	Double_t p_b[100];
	Double_t p_c[100];
	Double_t p_udsg[100];
	Double_t p_udscg[100];

	Double_t t_b[100];
	Double_t t_c[100];
	Double_t t_udsg[100];
	Double_t t_udscg[100];

	for (Int_t i=0;i<100;i++) {
		x_b[i] = CVS_b->Integral(i,100)/CVS_b->Integral(0,100) ;
		//  cout << " ud cut: "<< float(i)/100. << x_b[i] <<endl;
		x_c[i] = CVS_c->Integral(i,100)/CVS_c->Integral(0,100) ;
		x_udsg[i] = CVS_udsg->Integral(i,100)/CVS_udsg->Integral(0,100) ;
		x_udscg[i] = CVS_udscg->Integral(i,100)/CVS_udscg->Integral(0,100) ;

		p_b[i] = MyTag_b->Integral(i,100)/MyTag_b->Integral(0,100) ;
		p_c[i] = MyTag_c->Integral(i,100)/MyTag_c->Integral(0,100) ;
		p_udsg[i] = MyTag_udsg->Integral(i,100)/MyTag_udsg->Integral(0,100) ;
		p_udscg[i] = MyTag_udscg->Integral(i,100)/MyTag_udscg->Integral(0,100) ;

		t_b[i] = MyTag_b_test->Integral(i,100)/MyTag_b_test->Integral(0,100) ;

		t_c[i] = MyTag_c_test->Integral(i,100)/MyTag_c_test->Integral(0,100) ;

		t_udsg[i] = MyTag_udsg_test->Integral(i,100)/MyTag_udsg_test->Integral(0,100) ;

		t_udscg[i] = MyTag_udscg_test->Integral(i,100)/MyTag_udscg_test->Integral(0,100) ;
		//   if( t_b[i]>.6 && t_b[i]<.7 ) {
		//std::cout << "aMVa i "<< i << " "<< t_b[i] <<std::endl;
		// cout << "aMVa i "<< "c" << " "<< t_c[i] <<endl;
		//cout << "aMVa i "<< "u" << " "<< t_udsg[i] <<endl;
		// }
	}
	TCanvas* c1 = new TCanvas(myPlotFile+"graph1","A Simple Graph Example",200,10,700,500);

	TLegend* leg = new TLegend(0.2,0.7,0.4,0.9);
	TLegend* leg2 = new TLegend(0.675,0.15,0.925,0.4);

	c1->SetLogy();
	c1->SetGridy();
	c1->SetGridx();

	TGraph* gr_bc = new TGraph(100,x_b,x_c);
	gr_bc->GetXaxis()->SetTitle("b-tag efficiency");
	gr_bc->GetYaxis()->SetTitle("bkg-tag efficiency");
	gr_bc->GetYaxis()->SetTitle("bkg-tag efficiency");

	gr_bc->GetXaxis()->SetRangeUser(0.45,0.85);
	gr_bc->GetYaxis()->SetRangeUser(0.0005,0.8);
	/* gr_bc->GetXaxis()->SetTitleSize(0.05);
 gr_bc->GetXaxis()->SetTitleOffset(1.);
 gr_bc->GetYaxis()->SetTitleSize(0.05);
 gr_bc->GetYaxis()->SetTitleOffset(1.);
 gr_bc->SetTitle("");
	 */
	//leg->AddEntry( gr_bc, "b vs. c eff.","l");
	gr_bc->Draw("AC");

	TGraph* gr_budsg = new TGraph(100,x_b,x_udsg);
	gr_budsg->SetLineColor(kRed);
	gr_budsg->Draw("C");
	// TCanvas *c2 = new TCanvas("graph","A Simple Graph Example",200,10,700,500);
	leg->AddEntry( gr_bc, "1b vs. c","l");
	TGraph* gr_budscg = new TGraph(100,x_b,x_udscg);
	gr_budscg->SetLineColor(kBlue);
	gr_budscg->Draw("C");
	leg->AddEntry( gr_budsg, "1b vs. udsg","l");

	TGraph* pgr_bc = new TGraph(100,p_b,p_c);
	pgr_bc->SetLineStyle(2);
	pgr_bc->Draw("C");
	leg->AddEntry( gr_budscg, "1b vs. all","l");

	TGraph* pgr_budsg = new TGraph(100,p_b,p_udsg);
	pgr_budsg->SetLineStyle(2);
	pgr_budsg->SetLineColor(kRed);
	pgr_budsg->Draw("C");

	TGraph* pgr_budscg = new TGraph(100,p_b,p_udscg);
	pgr_budscg->SetLineStyle(2);
	pgr_budscg->SetLineColor(kBlue);
	pgr_budscg->Draw("C");

	TGraph* tgr_bc = new TGraph(100,t_b,t_c);
	tgr_bc->SetLineStyle(3);
	tgr_bc->Draw("C");

	TGraph* tgr_budsg = new TGraph(100,t_b,t_udsg);
	tgr_budsg->SetLineStyle(3);
	tgr_budsg->SetLineColor(kRed);
	tgr_budsg->Draw("C");

	TGraph* tgr_budscg = new TGraph(100,t_b,t_udscg);
	tgr_budscg->SetLineStyle(3);
	tgr_budscg->SetLineColor(kBlue);
	tgr_budscg->Draw("C");
	leg2->AddEntry( tgr_bc, "Deep CSV weights","l");
	leg2->AddEntry( gr_bc, "Deep cMVA trimmed weights","l");
	//if(CSV==true) leg2->AddEntry( pgr_bc, "CSV","l");
	//else  leg2->AddEntry( pgr_bc, "cMVA","l");
	leg->Draw("same");
	leg2->Draw("same");
	//TGraph* gr_bc = new TGraph(100,x_b,x_c);
	// gr_bc->Draw("AC*");
	c1->Write();
	c1->Print(myPlotFile+"_ROC.pdf");

	FileForPlots->Write();
	// FileForPlots->Delete();
}
