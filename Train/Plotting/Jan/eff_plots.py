

outdir='WP_plots'
import os
os.system('mkdir -p '+outdir)
outdir+='/'


infile='/afs/cern.ch/user/j/jkiesele/eos_DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_merged_PREDICTED/tree_association.txt'

dfudsprob='(prob_isUDS/(prob_isUDS + prob_isG))'

from testing import makeEffPlots_async

makeEffPlots_async(infile,      #input file or file list
                ['DF: tight WP','DF: medium WP','DF: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+'>0.9125',
                 dfudsprob+'>0.6675',
                 dfudsprob+'>0.5225'],
                'jet_pt>30 && jet_pt < 900 &&  isUD+isS', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_pt.pdf',  #output file (pdf)
                'jet p_{T} [GeV]',     #xaxisname
                'light quark efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise



makeEffPlots_async(infile,      #input file or file list
                ['DF: tight WP','DF: medium WP','DF: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+'>0.9125',
                 dfudsprob+'>0.6675',
                 dfudsprob+'>0.5225'],
                'jet_pt>30 && jet_pt < 900&&  isG', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_misID_jet_pt.pdf',  #output file (pdf)
                'jet p_{T} [GeV]',     #xaxisname
                'mis id probability' ,    #yaxisname
                rebinfactor=5 )  


makeEffPlots_async(infile,      #input file or file list
                ['DF: tight WP','DF: medium WP','DF: loose WP'],    #legend names (needs to be list)
                'jet_eta',    #variable to plot
                [dfudsprob+'>0.9125',
                 dfudsprob+'>0.6675',
                 dfudsprob+'>0.5225'],
                'jet_pt>30 && jet_pt < 900 &&  isUD+isS', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_eta.pdf',  #output file (pdf)
                'jet #eta',     #xaxisname
                'light quark efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise



makeEffPlots_async(infile,      #input file or file list
                ['DF: tight WP','DF: medium WP','DF: loose WP'],    #legend names (needs to be list)
                'jet_eta',    #variable to plot
                [dfudsprob+'>0.9125',
                 dfudsprob+'>0.6675',
                 dfudsprob+'>0.5225'],
                'jet_pt>30 && jet_pt < 900&&  isG', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_misID_jet_eta.pdf',  #output file (pdf)
                'jet #eta',     #xaxisname
                'mis id probability' ,    #yaxisname
                rebinfactor=5 )


makeEffPlots_async(infile,      #input file or file list
                ['DF: tight WP','DF: medium WP','DF: loose WP'],    #legend names (needs to be list)
                'npv',    #variable to plot
                [dfudsprob+'>0.9125',
                 dfudsprob+'>0.6675',
                 dfudsprob+'>0.5225'],
                'jet_pt>30 && jet_pt < 900 &&  isUD+isS', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_pileup.pdf',  #output file (pdf)
                'number of vertices',     #xaxisname
                'light quark efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise



makeEffPlots_async(infile,      #input file or file list
                ['DF: tight WP','DF: medium WP','DF: loose WP'],    #legend names (needs to be list)
                'npv',    #variable to plot
                [dfudsprob+'>0.9125',
                 dfudsprob+'>0.6675',
                 dfudsprob+'>0.5225'],
                'jet_pt>30 && jet_pt < 900&&  isG', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_misID_jet_pileup.pdf',  #output file (pdf)
                'number of vertices',     #xaxisname
                'mis id probability' ,    #yaxisname
                rebinfactor=5 ) 
