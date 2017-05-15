

from TrainData import TrainData, fileTimeOut
from TrainData import TrainData_simpleTruth
import numpy


class TrainData_test(TrainData_simpleTruth):
    '''
    class to show and test the density map
    '''
    
    def __init__(self):
        TrainData.__init__(self)
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
       
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles,createDensityMap
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        
        x_ch = createDensityMap(filename,TupleMeanStd,
                                   'Cpfcan_erel',
                                   self.nsamples,
                                   ['Cpfcan_eta','jet_eta',7,0.5],
                                   ['Cpfcan_phi','jet_phi',7,0.5],
                                   'nCpfcand',-1)
        x_neu = createDensityMap(filename,TupleMeanStd,
                                   'Npfcan_erel',
                                   self.nsamples,
                                   ['Npfcan_eta','jet_eta',7,0.5],
                                   ['Npfcan_phi','jet_phi',7,0.5],
                                   'nNpfcand',-1)
        x_sv = createDensityMap(filename,TupleMeanStd,
                                   'LooseIVF_sv_enratio',
                                   self.nsamples,
                                   ['LooseIVF_sv_eta','jet_eta',5,0.3],
                                   ['LooseIVF_sv_phi','jet_phi',5,0.3],
                                   'LooseIVF_nsv')
        
        for i in range(20):
            print(x_sv[i])
        
        self.w=[numpy.zeros(10)]
        self.x=[x_ch,x_neu]
        self.y=[numpy.zeros(10)]
        
        
        
        