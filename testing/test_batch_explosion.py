from DeepJetCore.training.training_base import training_base
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy
from DeepJetCore.evaluation import plotLoss



from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Model


class TrainData_forTest(TrainData):
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.truthclasses=['class1','class2']

        self.treename="deepntuplizer/tree"
        self.referenceclass='class1'
        
        
        self.registerBranches(self.truthclasses)
        self.registerBranches(['x'])
        
        self.weightbranchX='x'
        self.weightbranchY='x'
        
        self.weight_binX = numpy.array([-1,0.9,2.0],dtype=float)
        
        self.weight_binY = numpy.array(
            [-1,0.9,2.0],
            dtype=float
            )

        
             
        def reduceTruth(self, tuple_in):
        
            self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
            if tuple_in is not None:
                class1 = tuple_in['class1'].view(numpy.ndarray)
            
                class2 = tuple_in['class2'].view(numpy.ndarray)
                
                return numpy.vstack((class1,class2)).transpose()    
  
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd):
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
        
        truthtuple = Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
       
        newnsamp=x_all.shape[0]
        self.nsamples = newnsamp
        
        
        return x_all,alltruth

class TrainData_testingClass(TrainData_forTest):
   

    def __init__(self):
        TrainData_forTest.__init__(self)
        self.addBranches(['x'])

    def readFromRootFile(self,filename,TupleMeanStd,weighter):


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
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        weights=numpy.empty(self.nsamples)
        weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global]
        self.y=[alltruth]


def model_for_test(Inputs,nclasses,nregclasses):

    globalvars = (Inputs[0])
    
    x = Dense(50,activation='relu',kernel_initializer='lecun_uniform')(globalvars)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model







train=training_base(testrun=False)
newtraining= not train.modelSet()


if newtraining:
    train.setModel(model_for_test)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=5
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=100, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=1,
                                     plot_batch_loss = True)


