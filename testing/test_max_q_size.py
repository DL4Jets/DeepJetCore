from DeepJetCore.training.training_base import training_base
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy
from DeepJetCore.evaluation import plotLoss



from keras.callbacks import Callback
from keras.layers import Dense,BatchNormalization,Convolution1D
from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM,merge, Convolution1D, Conv2D
from keras.models import Model

class TrainDataDeepJet(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.undefTruth=['isUndefined']
        self.referenceclass='isB'
        self.truthclasses=['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC',
                           'isGCC','isUD','isS','isG','isUndefined']
        
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        
        self.weight_binX = numpy.array([
                10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves

class TrainData_fullTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose() 


class TrainData_deepFlavour_FT(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch
        
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


def block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x


def block_deepFlavourConvolutions(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def model_deepFlavourReference_test(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

train=training_base(testrun=False)
newtraining= not train.modelSet()


if newtraining:
    train.setModel(model_deepFlavourReference_test)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=5
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=3, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=150,
                                     plot_batch_loss = True)


