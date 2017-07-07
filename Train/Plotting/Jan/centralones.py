from testing import makeROCs_async

outdir='conv_rec'
import os
os.system('mkdir -p '+outdir)
outdir+='/'

files30_50=['/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_30_50_PRED/tree_association.txt',
            '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_30_50_PRED/tree_association.txt',
                       '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/recurrent_qcd_30_50/tree_association.txt',
                           '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/image_qcd_30_50/tree_association.txt']

files80_120=['/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_80_120_PRED/tree_association.txt',
             '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_80_120_PRED/tree_association.txt',
                        '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/recurrent_qcd_80_120/tree_association.txt',
                            '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/image_qcd_80_120/tree_association.txt']


files300_470=['/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_300_470_PRED/tree_association.txt',
              '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_300_470_PRED/tree_association.txt',
                         '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/recurrent_qcd_300_470/tree_association.txt',
                             '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/image_qcd_300_470/tree_association.txt']

files600_800=['/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_600_800_PRED/tree_association.txt',
              '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_600_800_PRED/tree_association.txt',
                         '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/recurrent_qcd_600_800/tree_association.txt',
                             '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/image_qcd_600_800/tree_association.txt']



legend= ['Q/G LH',
                 'DeepFlavour',
                 'recurrent',
                 'convolutional']

colors=['blue',
                 'purple',
                 'green,dashed',
                 'darkred,dotted']

probs=['jet_qgl',
                 '1.prob_isUDS/(1.prob_isUDS + 1.prob_isG)',
                 '2.prob_isUDS/(2.prob_isUDS + 2.prob_isG)',
                 '3.prob_isUDS/(3.prob_isUDS + 3.prob_isG)']

forward='((jet_eta>-2.4 && jet_eta<-1.3) || (jet_eta>1.3 && jet_eta<2.4))'
central='(jet_eta>-1.3 && jet_eta<1.3)'

makeROCs_async(intextfile=files30_50, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_30_50_forward.pdf", 
               cuts='jet_pt>30 &&'+forward, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 30-50 GeV', 
               secondcomment="jet p_{T} > 30 GeV, 1.3 < jet |#eta| < 2.4", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")

makeROCs_async(intextfile=files30_50, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_30_50_central.pdf", 
               cuts='jet_pt>30 &&'+central, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 30-50 GeV', 
               secondcomment="jet p_{T} > 30 GeV, jet |#eta| < 1.3", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")


makeROCs_async(intextfile=files80_120, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_80_120_forward.pdf", 
               cuts='jet_pt>70 &&'+forward, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 80-120 GeV', 
               secondcomment="jet p_{T} > 70 GeV, 1.3 < jet |#eta| < 2.4", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")



makeROCs_async(intextfile=files80_120, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_80_120_central.pdf", 
               cuts='jet_pt>70 &&'+central, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 80-120 GeV', 
               secondcomment="jet p_{T} > 70 GeV, jet |#eta| < 1.3", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")

makeROCs_async(intextfile=files300_470, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_300_470_forward.pdf", 
               cuts='jet_pt>250 &&'+forward, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 300-470 GeV', 
               secondcomment="jet p_{T} > 250 GeV, 1.3 < jet |#eta| < 2.4", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")



makeROCs_async(intextfile=files300_470, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_300_470_central.pdf", 
               cuts='jet_pt>250 &&'+central, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 300-470 GeV', 
               secondcomment="jet p_{T} > 250 GeV, jet |#eta| < 1.3", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")


makeROCs_async(intextfile=files600_800, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_600_800_forward.pdf", 
               cuts='jet_pt>500 &&'+forward, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 600-800 GeV', 
               secondcomment="jet p_{T} > 500 GeV, 1.3 < jet |#eta| < 2.4", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")



makeROCs_async(intextfile=files600_800, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_rec_conv_600_800_central.pdf", 
               cuts='jet_pt>500 &&'+central, 
               cmsstyle=True, 
               firstcomment='QCD events, #hat{p}_{T} = 600-800 GeV', 
               secondcomment="jet p_{T} > 500 GeV, jet |#eta| < 1.3", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")


exit()


makeROCs_async(intextfile=filesttbar, 
               name_list=legend, 
               probabilities_list=probs, 
               truths_list='isUD + isS', 
               vetos_list='isG', 
               colors_list=colors, 
               outpdffile=outdir+"ROC_QG_DF_yuta_ttbar_forward.pdf", 
               cuts='jet_pt>30 &&'+forward, 
               cmsstyle=True, 
               firstcomment='t#bar{t} events', 
               secondcomment="jet p_{T} > 30 GeV, 1.3 < jet |#eta| < 2.4", 
               invalidlist='', 
               extralegend=[], 
               logY=False, 
               individual=False,
               xaxis="light quark efficiency")


exit()

#makeROCs_async(intextfile=filesttbar, 
#               name_list=legend, 
#               probabilities_list=probs, 
#               truths_list='isUD + isS', 
#               vetos_list='isG', 
#               colors_list=colors, 
#               outpdffile=outdir+"ROC_QG_DF_yuta_ttbar_central.pdf", 
#               cuts='jet_pt>30 &&'+central, 
#               cmsstyle=True, 
#               firstcomment='t#bar{t} events', 
#               secondcomment="jet p_{T} > 30 GeV, jet |#eta| < 1.3", 
#               invalidlist='', 
#               extralegend=[], 
#               logY=False, 
#               individual=False,
#               xaxis="light quark efficiency")
#

