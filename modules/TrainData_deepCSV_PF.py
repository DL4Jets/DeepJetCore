'''
Created on 21 Feb 2017

@author: jkiesele
'''

from TrainData import TrainData
import numpy

class TrainData_deepCSV_PF(TrainData):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        self.truthclasses=['isB','isC','isUDS','isG']
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nsv'])
       
        self.addBranches(['Cpfcan_pt',
                              'Cpfcan_phirel',
                              'Cpfcan_etarel', 
                              'Cpfcan_dxy', 
                              'Cpfcan_dxyerr', 
                              'Cpfcan_dxysig', 
                              'Cpfcan_dz', 
                              'Cpfcan_VTX_ass', 
                              'Cpfcan_dptdpt', 
                              'Cpfcan_detadeta',
                              'Cpfcan_dphidphi',
                              'Cpfcan_dxydxy',
                              'Cpfcan_dzdz',
                              'Cpfcan_dxydz',
                              'Cpfcan_dphidxy',
                              'Cpfcan_dlambdadz',
                              #'Cpfcan_isMu',
                              #'Cpfcan_isEl',
                              'Cpfcan_chi2',
                              'Cpfcan_quality'
                              ],
                             20)
        
        
        self.addBranches(['Npfcan_pt',
                          'Npfcan_phirel',
                              'Npfcan_etarel',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              ],
                             15)
        
        
        self.addBranches(['sv_pt',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_chi2',
                              'sv_ndf',
                              'sv_dxy',
                              'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              ],
                             4)

        self.reducedtruthclasses=['isB','isC','isUDSG']
    
    def reduceTruth(self, tuple_in):
        b = tuple_in['isB'].view(numpy.ndarray)
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        l = g + uds
        self.reducedtruthclasses=['isB','isC','isUDSG']
        return numpy.vstack((b,c,l)).transpose()
       
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        self.fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        notremoves=weighter.createNotRemoveIndices(Tuple)
        
        print('took ', sw.getAndReset(), ' to create remove indices')
        
        weights=notremoves
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        
        print('remove')
        weights=weights[notremoves > 0]
        x_global=x_global[notremoves > 0]
        x_cpf=x_cpf[notremoves > 0]
        x_npf=x_npf[notremoves > 0]
        x_sv=x_sv[notremoves > 0]
        alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        
        
