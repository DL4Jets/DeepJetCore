


from testing import makeROCs_async, makePlots_async, testDescriptor
#from keras.models import load_model
from DataCollection import DataCollection


indir='/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/'
outdir='neutral_studies/'
import os
os.system('mkdir -p '+outdir)

trainings=['DF_FT_def',
           'DF_FT_noneut2',
           'DF_FT_lstm_neut']


trainings.extend(trainings)

filesttbar=[]
for t in trainings:
    filesttbar.append(indir+t+'/ttbar/tree_association.txt')
    
filesqcd=[]
for t in trainings:
    filesqcd.append(indir+t+'/qcd_600_800/tree_association.txt')
    

legend=['DeepFlavour^{*}','no neutrals','lstm neutr']

btruth='isB+isBB+isLeptonicB+isLeptonicB_C'
ctruth='isC'


bprob=[ '0.prob_isB+0.prob_isBB+0.prob_isLeptB',
        '1.prob_isB+1.prob_isBB+1.prob_isLeptB',
        '2.prob_isB+2.prob_isBB+2.prob_isLeptB',
                               
        '0.prob_isB+0.prob_isBB+0.prob_isLeptB',
        '1.prob_isB+1.prob_isBB+1.prob_isLeptB',
        '2.prob_isB+2.prob_isBB+2.prob_isLeptB',
        ]

cprob=[ '0.prob_isC',
        '1.prob_isC',
        '2.prob_isC',
        
        '0.prob_isC',
        '1.prob_isC',
        '2.prob_isC']

usdprob=['0.prob_isUDS',
         '1.prob_isUDS',
         '2.prob_isUDS',
                   
         '0.prob_isUDS',
         '1.prob_isUDS',
         '2.prob_isUDS',]




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
               logY=True)

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
               logY=True)
    
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
               logY=True)
    
    
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
               logY=True)

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
               logY=True)
        
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
               logY=False)    
