'''
Created on 21 Feb 2017

@author: jkiesele
'''

from TrainData import TrainData_Flavour
import numpy

class TrainData_deepCSV_PF(TrainData_Flavour):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.truthclasses=['isB','isC','isUDS','isG']
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand'])
       
        self.addBranches(['Cpfcan_phirel',
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
                             25)
        
        
        self.addBranches(['Npfcan_phirel',
                              'Npfcan_etarel',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              ],
                             20)
        
        self.addBranches(['nsv',
                              'sv_pt',
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
       
