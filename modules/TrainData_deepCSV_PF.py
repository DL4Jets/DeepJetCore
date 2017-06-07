'''
Created on 21 Feb 2017

@author: jkiesele
'''

from TrainData import TrainData, fileTimeOut
from TrainData import TrainData_simpleTruth
import numpy


class TrainData_deepCSV_softL_PF(TrainData_simpleTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv','softPFMuonBJetTags','softPFElectronBJetTags'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPtRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_erel',
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass'
                              ],
                             20)
        
        
        self.addBranches(['Npfcan_erel',
                          'Npfcan_deltaR',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              ],
                             15)
        
        
        self.addBranches(['sv_pt',
                              'sv_etarel',
                              'sv_phirel',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_chi2',
                              'sv_ndf',
                              'sv_normchi2',
                              'sv_dxy',
                              'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              'sv_enratio',
                              ],
                             4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
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
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
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
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        
        
    
class TrainData_deepCSV_softL_PF_reg(TrainData_simpleTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv','softPFMuonBJetTags','softPFElectronBJetTags'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPtRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_erel',
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass'
                              ],
                             20)
        
        
        self.addBranches(['Npfcan_erel',
                          'Npfcan_deltaR',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              ],
                             15)
        
        
        self.addBranches(['sv_pt',
                              'sv_etarel',
                              'sv_phirel',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_chi2',
                              'sv_ndf',
                              'sv_normchi2',
                              'sv_dxy',
                              'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              'sv_enratio',
                              ],
                             4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
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
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
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
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        
                


class TrainData_deepCSV_PF_rec(TrainData_simpleTruth):
    '''
    Add a recurrent LSTM layet instead of flattening, painfully slow, but more performant
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
   #     self.zeros_vectors([('nCpfcand',25),('nNpfcand',25),('nsv',4)])
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPtRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_erel',
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_erel',
                          'Npfcan_deltaR',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              
                              'Npfcan_puppiw'
                              ],
                             25)
        
        
        self.addBranches(['sv_pt',
                              'sv_etarel',
                              'sv_phirel',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_chi2',
                              'sv_ndf',
                              'sv_normchi2',
                              'sv_dxy',
                              'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              'sv_enratio',
                              ],
                             4)

        
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
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
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        mask = Tuple[['nCpfcand','nNpfcand','nsv']]
        maskListNpf = []
        maskListCpf = []
        maskListSv = []

        for i in range(0,Tuple.shape[0]):
            nMax = int(mask[i][0])
            if(nMax>25): nMax=25
            list0 = [[1.]*nMax+[0.]*(25-nMax)]*8
            nMax = int(mask[i][1])
            if(nMax>25): nMax=25
            maskListNpf.append(list0)

            nMax = int(mask[i][1])
            if(nMax>25): nMax=25
            list1 = [[1.]*nMax+[0.]*(25-nMax)]*4
            maskListCpf.append(list1)
            nMax = int(mask[i][2])
            if(nMax>4): nMax=4
            list2 = [[1.]*nMax+[0.]*(4-nMax)]*8
            maskListSv.append(list2)
            
        
        maskListNpf = numpy.asarray(maskListNpf,dtype=float)
        maskListCpf = numpy.asarray(maskListCpf,dtype=float)
        maskListSv = numpy.asarray(maskListSv,dtype=float)
        print ('zero shapes ', maskListNpf.shape, ' ' ,maskListCpf.shape  , ' ' ,maskListSv.shape )
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            maskListNpf = maskListNpf[notremoves > 0]
            maskListCpf = maskListCpf[notremoves > 0]
            maskListSv = maskListSv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        print (' types ',  type (x_cpf) , type (maskListNpf), ' ' ,type(maskListCpf) , ' ' , type(maskListSv) )
        self.x=[x_global,x_cpf,x_npf,x_sv,maskListNpf,maskListCpf,maskListSv]
        self.y=[alltruth]
        
        
class TrainData_deepCSV_PF(TrainData_simpleTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPtRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_erel',
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_erel',
                          'Npfcan_deltaR',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              
                              'Npfcan_puppiw'
                              ],
                             25)
        
        
        self.addBranches(['sv_pt',
                              'sv_etarel',
                              'sv_phirel',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_chi2',
                              'sv_ndf',
                              'sv_normchi2',
                              'sv_dxy',
                              'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              'sv_enratio',
                              ],
                             4)

        
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
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
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
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
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        
        
    
        

class TrainData_deepCSV_miniPF(TrainData_deepCSV_PF):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
       
        self.addBranches(['Cpfcan_pt',
                          'Cpfcan_ptrel',
                              'Cpfcan_deltaR', 
                              'Cpfcan_puppiw',
                              'Cpfcan_dxy', 
                              'Cpfcan_dxysig', 
                              'Cpfcan_fromPV', 
                              'Cpfcan_drminsv', 
                              
                              'Cpfcan_vertex_rho', 
                              'Cpfcan_vertex_phirel', 
                              'Cpfcan_vertex_etarel',
                              
                              'Cpfcan_chi2',
                              ],
                             10)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              ],
                             7)
        
        
        self.addBranches(['sv_pt',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_dxy',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3dsig',
                              'sv_enratio',
                              ],
                             4)
        
        #b-tag vars
        self.addBranches([ 'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
        
        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             7)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dVal', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)


        


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
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
        
        
        
        x_global_csv = MeanNormZeroPad(filename,TupleMeanStd,
                                   self.branches[4:],
                                   self.branchcutoffs[4:],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
        
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        


        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            x_global_csv=x_global_csv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,x_global_csv]
        self.y=[alltruth]
        
        

class TrainData_deepCSV_microPF(TrainData_deepCSV_PF):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta','nNpfcand',#'npv',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             6)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dVal', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)


        
        self.addBranches(['Npfcan_pt',
                          'Npfcan_ptrel',
                          'Npfcan_erel',
                          'Npfcan_phirel',
                          'Npfcan_etarel',
                          'Npfcan_deltaR',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              ],
                             15)


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   self.branches[0:4],
                                   self.branchcutoffs[0:4],self.nsamples)
        
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        
        
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
        
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_npf]
        self.y=[alltruth]
        
        
        
        
