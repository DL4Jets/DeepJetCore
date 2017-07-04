


from testing import makeROCs_async, makePlots_async, testDescriptor


outdir='jet_sel'

import os
os.system('mkdir -p '+outdir)
outdir+='/'


dir='/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_170_300/'
selectiontree=dir+'selectiontree.txt'
predictiontree=dir+'tree_association.txt'

makeROCs_async(intextfile=[predictiontree,selectiontree], 
               name_list=['DeepFlavour','Q/G LH'], 
               probabilities_list=['0.prob_isUDS/(0.prob_isG+0.prob_isUDS)','jet_qgl'], 
               truths_list='isUD+isS', 
               vetos_list='isG', 
               colors_list='auto', 
               outpdffile=outdir+"qcd_170_300.pdf", 
               cuts=2*['jet_pt>150']+2*['jet_pt>150 && keep && jet_no<3'], 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 170-300 GeV', 
               extralegend=['solid?all','dashed?selected'],
               secondcomment='jet p_{T} > 150 GeV',
               logY=False)

dir='/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_80_120/'
selectiontree=dir+'selectiontree.txt'
predictiontree=dir+'tree_association.txt'


makeROCs_async(intextfile=[predictiontree,selectiontree], 
               name_list=['DeepFlavour','Q/G LH'], 
               probabilities_list=['0.prob_isUDS/(0.prob_isG+0.prob_isUDS)','jet_qgl'], 
               truths_list='isUD+isS', 
               vetos_list='isG', 
               colors_list='auto', 
               outpdffile=outdir+"qcd_80_120.pdf", 
               cuts=2*['jet_pt>60']+2*['jet_pt>60 && keep && jet_no<3'], 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 80-120 GeV', 
               extralegend=['solid?all','dashed?selected'],
               secondcomment='jet p_{T} > 60 GeV',
               logY=False)


dir='/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_30_50_PREDICTED/'
selectiontree=dir+'selectiontree.txt'
predictiontree=dir+'tree_association.txt'


makeROCs_async(intextfile=[predictiontree,selectiontree], 
               name_list=['DeepFlavour','Q/G LH'], 
               probabilities_list=['0.prob_isUDS/(0.prob_isG+0.prob_isUDS)','jet_qgl'], 
               truths_list='isUD+isS', 
               vetos_list='isG', 
               colors_list='auto', 
               outpdffile=outdir+"qcd_30_50.pdf", 
               cuts=2*['jet_pt>30']+2*['jet_pt>30 && keep && jet_no<3'], 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 30-50 GeV', 
               extralegend=['solid?all','dashed?selected'],
               secondcomment='jet p_{T} > 30 GeV',
               logY=False)






exit()



