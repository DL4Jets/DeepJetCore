

from argparse import ArgumentParser

parser = ArgumentParser('make a set of ROC curves, comparing two training')
parser.add_argument('inputDirA')
parser.add_argument('inputDirB')
parser.add_argument('outputDir')
args = parser.parse_args()


outdir=args.outputDir+'/'


from testing import makeROCs_async, makePlots_async, testDescriptor
#from keras.models import load_model
from DataCollection import DataCollection

import os
os.system('mkdir -p '+outdir)

trainings=[args.inputDirA,
           args.inputDirB]


trainings.extend(trainings)

filesttbar=[]
for t in trainings:
    filesttbar.append(t+'/ttbar/tree_association.txt')
    
filesqcd=[]
for t in trainings:
    filesqcd.append(t+'/qcd_600_800/tree_association.txt')
    

legend=['standard','p_{T} cut']

btruth='isB+isBB+isGBB+isLeptonicB+isLeptonicB_C'
ctruth='isC+isCC+isGCC'


bprob=[ 'prob_isB+prob_isBB+prob_isLeptB',
        'prob_isB+prob_isBB+prob_isLeptB',
                               
        'prob_isB+prob_isBB+prob_isLeptB',
        'prob_isB+prob_isBB+prob_isLeptB',
        ]

cprob=[ 'prob_isC',
        'prob_isC',
        
        'prob_isC',
        'prob_isC']

usdprob=['prob_isUDS',
         'prob_isUDS',
                   
         'prob_isUDS',
         'prob_isUDS',]




print('creating ROCs')

#makeROCs_async(intextfile, 
#               name_list, 
#               probabilities_list, 
#               truths_list, 
#               vetos_list, 
#               colors_list, 
#               outpdffile, 
#               cuts, 
#               cmsstyle, 
#               firstcomment, 
#               secondcomment, 
#               extralegend+None, 
#               logY=True)


for ptcut in ['30','150']:
    
    makeROCs_async(intextfile=filesttbar, 
               name_list=legend, 
               probabilities_list=bprob, 
               truths_list=btruth, 
               vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*['isC'], 
               colors_list='auto', 
               outpdffile=outdir+"btag_pt"+ptcut+".pdf", 
               cuts='jet_pt>'+ptcut, 
               cmsstyle=True, 
               firstcomment='t#bar{t} events', 
               secondcomment='jet p_{T} > '+ptcut+' GeV', 
               extralegend=None, 
               logY=True,
               individual=True
               )

    makeROCs_async(intextfile=filesttbar, 
               name_list=legend, 
               probabilities_list=cprob, 
               truths_list=ctruth, 
               vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*[btruth], 
               colors_list='auto', 
               outpdffile=outdir+"ctag_pt"+ptcut+".pdf", 
               cuts='jet_pt>'+ptcut, 
               cmsstyle=True, 
               firstcomment='t#bar{t} events', 
               secondcomment='jet p_{T} > '+ptcut+' GeV', 
               extralegend=['solid?udsg','dashed?b'], 
               logY=True,
               individual=True)
    
    makeROCs_async(intextfile=filesttbar, 
               name_list=legend, 
               probabilities_list=usdprob, 
               truths_list='isUD+isS', 
               vetos_list=len(legend)*['isG']+len(legend)*['isB+isLeptonicB+isLeptonicB_C+isC'], 
               colors_list='auto', 
               outpdffile=outdir+"gtag_pt"+ptcut+".pdf", 
               cuts='jet_pt>'+ptcut, 
               cmsstyle=True, 
               firstcomment='t#bar{t} events', 
               secondcomment='jet p_{T} > '+ptcut+' GeV', 
               extralegend=['solid?g','dashed?bc'], 
               logY=True,
               individual=True)
    
    
makeROCs_async(intextfile=filesqcd, 
               name_list=legend, 
               probabilities_list=bprob, 
               truths_list=btruth, 
               vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*['isC'], 
               colors_list='auto', 
               outpdffile=outdir+"btag_qcd_pt400.pdf", 
               cuts='jet_pt>400', 
               cmsstyle=True, 
               firstcomment='QCD, 600 < p_{T} < 800 GeV', 
               secondcomment='jet p_{T} > 400 GeV', 
               extralegend=None, 
               logY=True,
               individual=True)

makeROCs_async(intextfile=filesqcd, 
               name_list=legend, 
               probabilities_list=cprob, 
               truths_list=ctruth, 
               vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*[btruth], 
               colors_list='auto', 
               outpdffile=outdir+"ctag_qcd_pt400.pdf", 
               cuts='jet_pt>400', 
               cmsstyle=True, 
               firstcomment='QCD, 600 < p_{T} < 800 GeV', 
               secondcomment='jet p_{T} > 400 GeV', 
               extralegend=['solid?udsg','dashed?b'], 
               logY=True,
               individual=True)
        
makeROCs_async(intextfile=filesqcd, 
               name_list=legend, 
               probabilities_list=usdprob, 
               truths_list='isUD+isS', 
               vetos_list=len(legend)*['isG']+len(legend)*['isB+isLeptonicB+isLeptonicB_C+isC'], 
               colors_list='auto', 
               outpdffile=outdir+"gtag_qcd_pt400.pdf", 
               cuts='jet_pt>400', 
               cmsstyle=True, 
               firstcomment='QCD, 600 < p_{T} < 800 GeV', 
               secondcomment='jet p_{T} > 400 GeV', 
               extralegend=['solid?g','dashed?bc'], 
               logY=False,
               individual=True)  


#individual plot for top/ttbar


makeROCs_async(intextfile=[filesttbar[1]], 
               name_list=['DeepFlavour'], 
               probabilities_list='prob_isUDS+prob_isC', 
               truths_list='isUD+isS+isC', 
               vetos_list=1*['isG']+1*['isB+isLeptonicB+isLeptonicB_C'], 
               colors_list='auto', 
               outpdffile=outdir+"lightQuarkJets_pt30.pdf", 
               cuts='jet_pt>400', 
               cmsstyle=True, 
               firstcomment='t#bar{t} events', 
               secondcomment='jet p_{T} > 30 GeV', 
               extralegend=['solid?g','dashed?b'], 
               logY=False,
               individual=True)  
  
