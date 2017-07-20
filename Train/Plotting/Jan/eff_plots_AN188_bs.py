

outdir='WP_plots'
import os
os.system('mkdir -p '+outdir)
outdir+='/'

from testing import makeEffPlots_async

dfudsprob='prob_isB+prob_isBB+prob_isLeptB'

infile='/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/ttbar/tree_association.txt'


TWP = '>0.825'
MWP = '>0.345'
LWP = '>0.055'

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>20 && (isB+isBB+isLeptonicB+isLeptonicB_C)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_pt.pdf',  #output file (pdf)
                'jet p_{T} [GeV]',     #xaxisname
                'j jet efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>20 && (isUD + isS + isG)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_pt_light.pdf',  #output file (pdf)
                'jet p_{T} [GeV]',     #xaxisname
                'light jet efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_eta',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30 && (isB+isBB+isLeptonicB+isLeptonicB_C)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_eta.pdf',  #output file (pdf)
                'jet #eta',     #xaxisname
                'b jet efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_eta',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30 && (isUD + isS + isG)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_eta_light.pdf',  #output file (pdf)
                'jet #eta',     #xaxisname
                'light jet efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'npv',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30 && (isB+isBB+isLeptonicB+isLeptonicB_C)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_npv.pdf',  #output file (pdf)
                'npv',     #xaxisname
                'b jet efficiency' ,    #yaxisname
                   rebinfactor=5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30 && (isUD + isS + isG)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_npv_light.pdf',  #output file (pdf)
                'npv',     #xaxisname
                'light jet efficiency' ,    #yaxisname
                   rebinfactor=5)       #normalise



infile='/afs/cern.ch/user/j/jkiesele/eos_DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_merged_PREDICTED/tree_association.txt'


makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>20 && (isB+isBB+isLeptonicB+isLeptonicB_C)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_pt_QCD.pdf',  #output file (pdf)
                'jet p_{T} [GeV]',     #xaxisname
                'b jet efficiency' ,    #yaxisname
                   rebinfactor=1, Xmin=20., Xmax=900.)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'jet_pt',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>20 && (isUD + isS + isG)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_pt_light_QCD.pdf',  #output file (pdf)
                'jet p_{T} [GeV]',     #xaxisname
                'light jet efficiency' ,    #yaxisname
                   rebinfactor=1, SetLogY = True, minimum = 0.0005, maximum = 1.5, Xmin=20., Xmax=900.)       #normalise


makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'fabs(jet_eta)',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30 && jet_pt <150 && (isB+isBB+isLeptonicB+isLeptonicB_C)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_eta_QCD.pdf',  #output file (pdf)
                'jet #eta',     #xaxisname
                'b jet efficiency' ,    #yaxisname
                 rebinfactor=5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'fabs(jet_eta)',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30 && jet_pt <150 && (isUD + isS + isG)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_eta_light_QCD.pdf',  #output file (pdf)
                'jet #eta',     #xaxisname
                'light jet efficiency' ,    #yaxisname
                   rebinfactor=5, SetLogY = True, minimum = 0.0005, maximum = 1.5)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'npv',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30  && jet_pt <150 && (isB+isBB+isLeptonicB+isLeptonicB_C)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_npv_QCD.pdf',  #output file (pdf)
                'npv',     #xaxisname
                'b jet efficiency' ,    #yaxisname
                   rebinfactor=5, Xmin=9, Xmax=65)       #normalise

makeEffPlots_async(infile,      #input file or file list
                ['DJ: tight WP','DJ: medium WP','DJ: loose WP'],    #legend names (needs to be list)
                'npv',    #variable to plot
                [dfudsprob+TWP,
                 dfudsprob+MWP,
                 dfudsprob+LWP],
                'jet_pt>30  && jet_pt <150  &&(isUD + isS + isG)', #cut to apply
                'auto',     #line color and style (e.g. 'red,dashed')
                outdir+'DF_ID_jet_npv_light_QCD.pdf',  #output file (pdf)
                'npv',     #xaxisname
                'light jet efficiency' ,    #yaxisname
                   rebinfactor=5, SetLogY = True, minimum = 0.0005, maximum = 1.5, Xmin=9, Xmax=65)       #normalise

