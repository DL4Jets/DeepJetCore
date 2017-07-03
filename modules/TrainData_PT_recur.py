from TrainData import TrainData_fullTruth, TrainData_quarkGluon
from TrainData import TrainData,fileTimeOut
from keras.layers import Dense, Dropout, LSTM
from keras.layers.merge import concatenate
from keras.models import Model

class TrainData_QG_simple(TrainData_fullTruth):
    def __init__(self):
        super(TrainData_QG_simple, self).__init__()
        self.addBranches(['jet_pt', 'jet_eta', 'rho', 'QG_ptD', 'QG_axis2', 'QG_mult'])

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormZeroPad
        import numpy
        from stopwatch import stopwatch
        import c_meanNormZeroPad
        c_meanNormZeroPad.zeroPad()
        
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
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        nparray = self.readTreeFromRootToTuple(filename)        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(nparray)
            undef=nparray['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(nparray)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.ones(self.nsamples)
        
        truthtuple =  nparray[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)

        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
                        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        self.w=[weights]
        self.x=[x_global]
        self.y=[alltruth]
        
    @staticmethod
    def model(input_shapes, nclasses):
        from keras.layers import Input
        from keras.layers.core import Masking
        x_global  = Input(shape=input_shapes[0])
        x = Dense(10, activation='relu',kernel_initializer='lecun_uniform')(x_global)
        for _ in range(6):
            x = Dense(10, activation='relu',kernel_initializer='lecun_uniform')(x)

        predictions = Dense(
            nclasses, activation='softmax',
            kernel_initializer='lecun_uniform',
            name='classification_out'
        )(x)
        return Model(inputs=x_global, outputs=predictions)

class TrainData_PT_recur(TrainData_quarkGluon):#TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_PT_recur, self).__init__()
        self.regressiontargetclasses=['uncPt','Pt']        
        
        self.addBranches([
            #base
            'jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','rho',
            #q/g enhancements
            #'QG_ptD', 'QG_axis2', 'QG_mult',
        ])
       
        self.addBranches([
            'Cpfcan_ptrel', #not the same as btv ptrel!
            #'Cpfcan_erel',
            'Cpfcan_phirel',
            'Cpfcan_etarel',
            #'Cpfcan_pt', 
            'Cpfcan_puppiw',
            #'Cpfcan_quality'
        ], 25)
        
        
        self.addBranches([
            'Npfcan_ptrel', #not the same as btv ptrel!
            #'Cpfcan_erel',
            'Npfcan_phirel',
            'Npfcan_etarel',
            #'Npfcan_pt', 
            'Npfcan_puppiw',
            #'Npfcan_quality'
        ], 25)
        
        self.regtruth='gen_pt_WithNu'
        self.regreco='jet_corr_pt'
        self.registerBranches([self.regtruth, self.regreco])
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        import c_meanNormZeroPad
        c_meanNormZeroPad.zeroPad()
        
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
        
     
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        nparray = self.readTreeFromRootToTuple(filename)        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(nparray)
            undef=nparray['isUndefined']
            np_slice = nparray[[
                    'isB', 'isBB',
                    'isLeptonicB', 'isLeptonicB_C',
                    'isC',
                    ]]
            np_slice.dtype = numpy.dtype('<u4') #flatten and remove records
            np_slice = np_slice.reshape(nparray.shape + (-1,))
            hf = np_slice.any(axis=1)
            notremoves -= undef
            notremoves -= hf
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(nparray)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.ones(self.nsamples)
        
        pttruth = nparray[self.regtruth]
        ptreco  = nparray[self.regreco]        
        truthtuple =  nparray[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #
        # sort vectors (according to pt at the moment)
        #
        idxs = x_cpf[:,:,0].argsort() #0 is pt ratio
        xshape = x_cpf.shape
        static_idxs = numpy.indices(xshape)
        idxs = idxs.reshape((xshape[0], xshape[1], 1))
        idxs = numpy.repeat(idxs, xshape[2], axis=2)
        x_cpf = x_cpf[static_idxs[0], idxs, static_idxs[2]]

        idxs = x_npf[:,:,0].argsort() #0 is pt ratio
        xshape = x_npf.shape
        static_idxs = numpy.indices(xshape)
        idxs = idxs.reshape((xshape[0], xshape[1], 1))
        idxs = numpy.repeat(idxs, xshape[2], axis=2)
        x_npf = x_npf[static_idxs[0], idxs, static_idxs[2]]

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            x_npf = x_npf[notremoves > 0]
           # x_npf=x_npf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            pttruth=pttruth[notremoves > 0]
            ptreco=ptreco[notremoves > 0]
                        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,ptreco]
        self.y=[alltruth,pttruth]
        
    @staticmethod
    def base_model(input_shapes):
        from keras.layers import Input
        from keras.layers.core import Masking
        x_global  = Input(shape=input_shapes[0])
        x_charged = Input(shape=input_shapes[1])
        x_neutral = Input(shape=input_shapes[2])
        x_ptreco  = Input(shape=input_shapes[3])
        lstm_c = Masking()(x_charged)
        lstm_c = LSTM(100,go_backwards=True,implementation=2)(lstm_c)
        lstm_n = Masking()(x_neutral)
        lstm_n = LSTM(100,go_backwards=True,implementation=2)(lstm_n)
        x = concatenate( [lstm_c, lstm_n, x_global] )
        x = Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
        x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
        x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
        x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
        x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
        x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
        x = concatenate([x, x_ptreco])
        return [x_global, x_charged, x_neutral, x_ptreco], x

    @staticmethod
    def regression_generator(generator):
        for X, Y in generator:
            yield X, Y[1]#.astype(int)

    @staticmethod
    def regression_model(input_shapes):
        inputs, x = TrainData_PT_recur.base_model(input_shapes)
        predictions = Dense(
            2, activation='linear', 
            init='normal', name='regression_out')(x)
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def mse_regression_model(input_shapes):
        inputs, x = TrainData_PT_recur.base_model(input_shapes)
        predictions = Dense(
            1, activation='linear',
            init='normal', name='regression_out')(x)
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def classification_generator(generator):
        for X, Y in generator:
            yield X, Y[0]#.astype(int)

    @staticmethod
    def classification_model(input_shapes, nclasses):
        inputs, x = TrainData_PT_recur.base_model(input_shapes)
        predictions = Dense(
            nclasses, activation='softmax',
            kernel_initializer='lecun_uniform',
            name='classification_out'
        )(x)
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def model(input_shapes, nclasses):
        inputs, x = TrainData_PT_recur.base_model(input_shapes)
        predictions = [
            Dense(
                nclasses, 
                activation='softmax', kernel_initializer='lecun_uniform',
                name='classification_out'
            )(x),
            Dense(2, activation='linear', 
                  init='normal', name='regression_out')(x),
        ]
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def mse_model(input_shapes, nclasses):
        inputs, x = TrainData_PT_recur.base_model(input_shapes)
        predictions = [
            Dense(
                nclasses, 
                activation='softmax', kernel_initializer='lecun_uniform',
                name='classification_out'
            )(x),
            Dense(1, activation='linear', 
                  init='normal', name='regression_out')(x),
        ]
        return Model(inputs=inputs, outputs=predictions)



class TrainData_recurrent_fullTruth(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_recurrent_fullTruth, self).__init__()
        self.regressiontargetclasses=['uncPt','Pt']
        
        self.addBranches([
            #base
            'jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','rho',
            #q/g enhancements
            #'QG_ptD', 'QG_axis2', 'QG_mult',
        ])
       
        self.addBranches([
            'Cpfcan_ptrel', #not the same as btv ptrel!
            #'Cpfcan_erel',
            'Cpfcan_phirel',
            'Cpfcan_etarel',
            #'Cpfcan_pt', 
            'Cpfcan_puppiw',
            #'Cpfcan_quality'
        ], 25)
        
        
        self.addBranches([
            'Npfcan_ptrel', #not the same as btv ptrel!
            #'Cpfcan_erel',
            'Npfcan_phirel',
            'Npfcan_etarel',
            #'Npfcan_pt', 
            'Npfcan_puppiw',
            #'Npfcan_quality'
        ], 25)
        
        self.regtruth='gen_pt_WithNu'
        self.regreco='jet_corr_pt'
        self.registerBranches([self.regtruth, self.regreco])
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        import c_meanNormZeroPad
        c_meanNormZeroPad.zeroPad()
        
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
        
     
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        nparray = self.readTreeFromRootToTuple(filename)        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(nparray)
            undef=nparray['isUndefined']
            notremoves -= undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(nparray)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.ones(self.nsamples)
        
        pttruth = nparray[self.regtruth]
        ptreco  = nparray[self.regreco]        
        truthtuple =  nparray[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #
        # sort vectors (according to pt at the moment)
        #
        idxs = x_cpf[:,:,0].argsort() #0 is pt ratio
        xshape = x_cpf.shape
        static_idxs = numpy.indices(xshape)
        idxs = idxs.reshape((xshape[0], xshape[1], 1))
        idxs = numpy.repeat(idxs, xshape[2], axis=2)
        x_cpf = x_cpf[static_idxs[0], idxs, static_idxs[2]]

        idxs = x_npf[:,:,0].argsort() #0 is pt ratio
        xshape = x_npf.shape
        static_idxs = numpy.indices(xshape)
        idxs = idxs.reshape((xshape[0], xshape[1], 1))
        idxs = numpy.repeat(idxs, xshape[2], axis=2)
        x_npf = x_npf[static_idxs[0], idxs, static_idxs[2]]

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            x_npf = x_npf[notremoves > 0]
           # x_npf=x_npf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            pttruth=pttruth[notremoves > 0]
            ptreco=ptreco[notremoves > 0]
                        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,ptreco]
        self.y=[alltruth,pttruth]






