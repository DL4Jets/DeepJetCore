from testing import makeROCs_async

btruth='isB+isBB+isLeptonicB+isLeptonicB_C'

# b vs (udsg or c) [ttbar]

makeROCs_async( ['/eos/cms/store/cmst3/group/dehep/DeepJet/Anna_DeepCSV/ttbar_CFR_PRED/tree_association.txt',
                 '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/ttbar/tree_association.txt'],
                ['DeepCSV',
                 'DeepFlavour'],
                ['prob_isC/(prob_isC + prob_isUDSG)',
                 'prob_isC/(prob_isC + prob_isUDS + prob_isG)'],
                ['isC',
                 'isC'],
                ['isUD + isS + isG',
                 'isUD + isS + isG'],
                ['blue',
                 'purple'],
                 "ROC_C_ttbar_binary.pdf",
                 'jet_pt>30 && jet_eta>0',
                 True,
                 'ttbar events',
#                 "#splitline{150 < jet p_{T} < 300  [GeV]}{jet |#eta| < 2.4}",
                 "jet p_{T} > 30 GeV",
                 '',
                 [],
                 True,
                 True,
                 "C efficiency"
                 )


makeROCs_async( ['/eos/cms/store/cmst3/group/dehep/DeepJet/Anna_DeepCSV/ttbar_CFR_PRED/tree_association.txt',
                 '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/ttbar/tree_association.txt'],
                ['DeepCSV',
                 'DeepFlavour'],
                ['prob_isC',
                 'prob_isC'],
                ['isC',
                 'isC'],
                ['isUD + isS + isG',
                 'isUD + isS + isG'],
                ['blue',
                 'purple'],
                 "ROC_C_ttbar.pdf",
                 'jet_pt>30 && jet_eta>0',
                 True,
                 'ttbar events',
#                 "#splitline{150 < jet p_{T} < 300  [GeV]}{jet |#eta| < 2.4}",
                 "jet p_{T} > 30 GeV",
                 '',
                 [],
                 True,
                 True,
                 "C efficiency"
                 )





# c vs udsg [ttbar]

makeROCs_async( ['/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/ttbar/tree_association.txt',
                 '/eos/cms/store/cmst3/group/dehep/DeepJet/Anna_DeepCSV/ttbar_CFR_PRED/tree_association.txt',
                 '/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/ttbar/tree_association.txt'],
                ['CSVv2',
                 'DeepCSV',
                 'DeepFlavour'],
                ['pfCombinedInclusiveSecondaryVertexV2BJetTags',
                 'prob_isB + prob_isBB',
                 'prob_isB + prob_isBB + prob_isLeptB',
                 'pfCombinedInclusiveSecondaryVertexV2BJetTags',
                 'prob_isB + prob_isBB',
                 'prob_isB + prob_isBB + prob_isLeptB'],
                [btruth,
                 btruth,
                 btruth,
                 btruth,
                 btruth,
                 btruth],
                ['isUD + isS + isG',
                 'isUD + isS + isG',
                 'isUD + isS + isG',
                 'isC',
                 'isC',
                 'isC'],
                ['green',
                 'blue',
                 'purple',
                 'green,dashed',
                 'blue,dashed',
                 'purple,dashed'],
                 "ROC_ttbar.pdf",
                 'jet_pt>30 && jet_eta>0',
                 True,
                 'ttbar events',
#                 "#splitline{150 < jet p_{T} < 300  [GeV]}{jet |#eta| < 2.4}",
                 "jet p_{T} > 30 GeV",
                 '',
                 ['solid?L','dashed?C'],
                 True,
                 True,
                 "B efficiency"
                 )
