


btruth='isB+isBB+isLeptonicB+isLeptonicB_C'


makeROCs_async('path_to_you_prediction/tree_association.txt',
                                   ['B vs light',
                                   'B vs C'],
         
          ['prob_isB+prob_isBB',
           'prob_isB+prob_isBB'],
           btruth,
         ['isUD+isS','isC'],
         ['blue',
          'purple'],
         outputDir+"b_tag_pt30.pdf",'jet_pt>30 '
         ,False,
         '',
         '')