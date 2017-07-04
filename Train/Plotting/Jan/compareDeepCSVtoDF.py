


from testing import makeROCs_async, testDescriptor
#from keras.models import load_model
from DataCollection import DataCollection
import os

indir='/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/'
outdir='compareDeepCSVtoDF/'
os.system('mkdir -p '+outdir)

trainings=['deepCSV_conv',
           'deepCSV_conv_more',
           'deepCSV_conv_more_more',
           'deepCSV_conv_more_more_more']

trainings.extend(trainings)

filesttbar=[]
for t in trainings:
    filesttbar.append(indir+t+'/ttbar/tree_association.txt')
    
filesqcd=[]
for t in trainings:
    filesqcd.append(indir+t+'/qcd_600_800/tree_association.txt')
    

legend=['deepCSV^{+}','+DF tracks','+DF neutrals','+DF SV']

btruth='isB+isBB+isLeptonicB+isLeptonicB_C'
ctruth='isC'


bprob=[ '0.prob_isB+0.prob_isBB',
        '1.prob_isB+1.prob_isBB',
        '2.prob_isB+2.prob_isBB',
        '3.prob_isB+3.prob_isBB',
        
        '0.prob_isB+0.prob_isBB',
        '1.prob_isB+1.prob_isBB',
        '2.prob_isB+2.prob_isBB',
        '3.prob_isB+3.prob_isBB',
        ]

cprob=[ '0.prob_isC',
        '1.prob_isC',
        '2.prob_isC',
        '3.prob_isC',
        
        '0.prob_isC',
        '1.prob_isC',
        '2.prob_isC',
        '3.prob_isC']

lprob='prob_isUDSG'


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
#               invalidlist, 
#               extralegend, 
#               logY)
#

for ptcut in ['30','150']:
    
    
    makeROCs_async(intextfile=filesttbar,
                   name_list=legend,
                   probabilities_list=bprob,
                   truths_list=btruth, #signal truth
                   vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*[ctruth],
                   colors_list='auto',
                   outpdffile=outdir+"btag_pt"+ptcut+".pdf",
                   cuts='jet_pt>'+ptcut,
                   cmsstyle=True,
                   firstcomment='t#bar{t} events',
                   secondcomment='jet p_{T} > '+ptcut+' GeV') 
    
    makeROCs_async(intextfile=filesttbar,
                   name_list=legend,
                   probabilities_list=cprob,
                   truths_list=ctruth, #signal truth
                   vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*[btruth],
                   colors_list='auto',
                   outpdffile=outdir+"ctag_pt"+ptcut+".pdf",
                   cuts='jet_pt>'+ptcut,
                   cmsstyle=True,
                   firstcomment='t#bar{t} events',
                   secondcomment='jet p_{T} > '+ptcut+' GeV',
                   extralegend=['solid?udsg','dashed?b'])
     
    
    
  
makeROCs_async(intextfile=filesqcd,
               name_list=legend,
               probabilities_list=bprob,
               truths_list=btruth, #signal truth
               vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*[ctruth],
               colors_list='auto',
               outpdffile=outdir+"btag_qcd_pt400.pdf",
               cuts='jet_pt>400',
               cmsstyle=True,
               firstcomment='QCD, 600 < p_{T} < 800 GeV',
               secondcomment='jet p_{T} > 400 GeV') 


    
makeROCs_async(intextfile=filesqcd,
               name_list=legend,
               probabilities_list=cprob,
               truths_list=ctruth, #signal truth
               vetos_list=len(legend)*['isUD+isS+isG']+len(legend)*[btruth],
               colors_list='auto',
               outpdffile=outdir+"ctag_qcd_pt400.pdf",
               cuts='jet_pt>400',
               cmsstyle=True,
               firstcomment='QCD, 600 < p_{T} < 800 GeV',
               secondcomment='jet p_{T} > 400 GeV',
               extralegend=['solid?udsg','dashed?b'],
               logY=False)  

