


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
from pdb import set_trace
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import multi_gpu_model
from _thread import start_new_thread

import imp
try:
    imp.find_module('Losses')
    from Losses import *
except ImportError:
    print('No Losses module found, ignoring at your own risk')
    global_loss_list = {}

try:
    imp.find_module('Layers')
    from Layers import *
except ImportError:
    print('No Layers module found, ignoring at your own risk')
    global_layers_list = {}

try:
    imp.find_module('Metrics')
    from Metrics import *
except ImportError:
    print('No metrics module found, ignoring at your own risk')
    global_metrics_list = {}
custom_objects_list = {}
custom_objects_list.update(djc_global_loss_list)
custom_objects_list.update(djc_global_layers_list)
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

##helper

from keras.models import Model
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load save and predict methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname or 'predict' in attrname:
            return getattr(self._smodel, attrname)
        else:
            #return Model.__getattribute__(self, attrname)
            return super(ModelMGPU, self).__getattribute__(attrname)



class training_base(object):
    
    def __init__(
				self, splittrainandtest=0.85,
				useweights=False, testrun=False,
                testrun_fraction=0.1,
				resumeSilently=False, 
				renewtokens=False,
				collection_class=DataCollection,
				parser=None,
                recreate_silently=False
				):
        
        import sys
        scriptname=sys.argv[0]
        
        if parser is None: parser = ArgumentParser('Run the training')
        parser.add_argument('inputDataCollection')
        parser.add_argument('outputDir')
        parser.add_argument('--modelMethod', help='Method to be used to instantiate model in derived training class', metavar='OPT', default=None)
        parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default=-1)
        parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
        parser.add_argument("--submitbatch",  help="submits the job to condor" , default=False, action="store_true")
        parser.add_argument("--walltime",  help="sets the wall time for the batch job, format: 1d5h or 2d or 3h etc" , default='1d')
        parser.add_argument("--isbatchrun",   help="is batch run", default=False, action="store_true")
        
        
        args = parser.parse_args()
        self.args = args
        import sys
        self.argstring = sys.argv
        #sanity check
        if args.isbatchrun:
            args.submitbatch=False
            resumeSilently=True
            
        if args.submitbatch:
            print('submitting batch job. Model will be compiled for testing before submission (GPU settings being ignored)')
        
        
        import matplotlib
        #if no X11 use below
        matplotlib.use('Agg')
        if args.gpu<0:
            import imp
            try:
                imp.find_module('setGPU')
                import setGPU
            except :
                found = False
        else:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            print('running on GPU(s) '+str(args.gpu))
        
        if args.gpufraction>0 and args.gpufraction<1:
            import sys
            import tensorflow as tf
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            import keras
            from keras import backend as K
            K.set_session(sess)
            print('using gpu memory fraction: '+str(args.gpufraction))
        
            
            
        
        import keras
                
        self.ngpus=1
        if (not args.gpu<0) and len(args.gpu):
            self.ngpus=len([i for i in args.gpu.split(',')])
            print('running on '+str(self.ngpus)+ ' gpus')
            
        self.keras_inputs=[]
        self.keras_inputsshapes=[]
        self.keras_model=None
        self.keras_model_method=args.modelMethod
        self.train_data=None
        self.val_data=None
        self.startlearningrate=None
        self.optimizer=None
        self.trainedepoches=0
        self.compiled=False
        self.checkpointcounter=0
        self.renewtokens=renewtokens
        if args.isbatchrun:
            self.renewtokens=False
        self.callbacks=None
        self.custom_optimizer=False
        self.copied_script=""
        self.submitbatch=args.submitbatch
        
        self.GAN_mode=False
        
        self.inputData = os.path.abspath(args.inputDataCollection) \
												 if ',' not in args.inputDataCollection else \
														[os.path.abspath(i) for i in args.inputDataCollection.split(',')]
        self.outputDir=args.outputDir
        # create output dir
        
        isNewTraining=True
        if os.path.isdir(self.outputDir):
            if not (resumeSilently or recreate_silently):
                var = raw_input('output dir exists. To recover a training, please type "yes"\n')
                if not var == 'yes':
                    raise Exception('output directory must not exists yet')
            isNewTraining=False
            if recreate_silently:
                isNewTraining=True     
        else:
            os.mkdir(self.outputDir)
        self.outputDir = os.path.abspath(self.outputDir)
        self.outputDir+='/'
        
        if recreate_silently:
            os.system('rm -rf '+ self.outputDir +'*')
        
        #copy configuration to output dir
        if not args.isbatchrun:
            shutil.copyfile(scriptname,self.outputDir+os.path.basename(scriptname))
            self.copied_script = self.outputDir+os.path.basename(scriptname)
        else:
            self.copied_script = scriptname
        
        self.train_data = collection_class()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights=useweights
        
        if testrun:
            self.train_data.split(testrun_fraction)
            self.val_data=self.train_data
        else:    
            self.val_data=self.train_data.split(splittrainandtest)
        


        shapes=self.train_data.getKerasFeatureShapes()
        print("shapes", shapes)
        
        self.keras_inputs=[]
        self.keras_inputsshapes=[]
        
        for s in shapes:
            self.keras_inputs.append(keras.layers.Input(shape=s))
            self.keras_inputsshapes.append(s)
            
        if not isNewTraining:
            kfile = self.outputDir+'/KERAS_check_model_last.h5' \
							 if os.path.isfile(self.outputDir+'/KERAS_check_model_last.h5') else \
							 self.outputDir+'/KERAS_model.h5'
            if os.path.isfile(kfile):
                self.loadModel(kfile)
                self.trainedepoches=sum(1 for line in open(self.outputDir+'losses.log'))
            else:
                print('no model found in existing output dir, starting training from scratch')
        
        
    def __del__(self):
        if hasattr(self, 'train_data'):
            del self.train_data
            del self.val_data
        
    def modelSet(self):
        return not self.keras_model==None
        
    def setModel(self,model,**modelargs):
        if len(self.keras_inputs)<1:
            raise Exception('setup data first') 
        try:
            self.keras_model=model(self.keras_inputs,**modelargs)
        except BaseException as e:
            print('problem in setting model. Reminder: since DJC 2.0, NClassificationTargets and RegressionTargets must not be specified anymore')
            raise e
        if not self.keras_model:
            raise Exception('Setting model not successful') 
        
    
    def saveCheckPoint(self,addstring=''):
        
        self.checkpointcounter=self.checkpointcounter+1 
        self.saveModel("KERAS_model_checkpoint_"+str(self.checkpointcounter)+"_"+addstring +".h5")    
           
        
    def loadModel(self,filename):
        from keras.models import load_model
        self.keras_model=load_model(filename, custom_objects=custom_objects_list)
        self.optimizer=self.keras_model.optimizer
        self.compiled=True
        if self.ngpus>1:
            self.compiled=False
        
    def setCustomOptimizer(self,optimizer):
        self.optimizer = optimizer
        self.custom_optimizer=True
        
    def compileModel(self,
                     learningrate,
                     clipnorm=None,
                     discriminator_loss=['binary_crossentropy'],
                     print_models=False,
                     metrics=None,
                     **compileargs):
        if not self.keras_model and not self.GAN_mode:
            raise Exception('set model first') 

        if self.ngpus>1 and not self.submitbatch:
            print('Model being compiled for '+str(self.ngpus)+' gpus')
            self.keras_model = ModelMGPU(self.keras_model, gpus=self.ngpus)
            
        self.startlearningrate=learningrate
        
        if not self.custom_optimizer:
            from keras.optimizers import Adam
            if clipnorm:
                self.optimizer = Adam(lr=self.startlearningrate,clipnorm=clipnorm)
            else:
                self.optimizer = Adam(lr=self.startlearningrate)
            
            
         
        self.keras_model.compile(optimizer=self.optimizer,metrics=metrics,**compileargs)
        if print_models:
            print(self.keras_model.summary())
        self.compiled=True

    def compileModelWithCustomOptimizer(self,
                                        customOptimizer,
                                        **compileargs):
        raise Exception('DEPRECATED: please use setCustomOptimizer before calling compileModel') 
        
        
    def saveModel(self,outfile):
        if not self.GAN_mode:
            self.keras_model.save(self.outputDir+outfile)
        else:
            self.gan.save(self.outputDir+'GAN_'+outfile)
            self.generator.save(self.outputDir+'GEN_'+outfile)
            self.discriminator.save(self.outputDir+'DIS_'+outfile)
        
        
    def _initTraining(self,
                      nepochs,
                     batchsize,
                     use_sum_of_squares=False):
        
        
        if self.submitbatch:
            from DeepJetCore.training.batchTools import submit_batch
            submit_batch(self, self.args.walltime)
            exit() #don't delete this!
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.batch_uses_sum_of_squares=use_sum_of_squares
        self.val_data.batch_uses_sum_of_squares=use_sum_of_squares
        
        self.train_data.writeToFile(self.outputDir+'trainsamples.djcdc')
        self.val_data.writeToFile(self.outputDir+'valsamples.djcdc')
        
        #make sure tokens don't expire
        from .tokenTools import checkTokens, renew_token_process
        
        if self.renewtokens:
            print('starting afs backgrounder')
            checkTokens()
            start_new_thread(renew_token_process,())
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        
        
    def trainModel(self,
                   nepochs,
                   batchsize,
                   batchsize_use_sum_of_squares = False,
                   stop_patience=-1, 
                   lr_factor=0.5,
                   lr_patience=-1, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   checkperiod=10,
                   additional_plots=None,
                   additional_callbacks=None,
                   load_in_mem = False,
                   plot_batch_loss = False,
                   **trainargs):
        
        
        
        # write only after the output classes have been added
        self._initTraining(nepochs,batchsize, batchsize_use_sum_of_squares)
        
        #self.keras_model.save(self.outputDir+'KERAS_check_last_model.h5')
        print('setting up callbacks')
        from .DeepJet_callbacks import DeepJet_callbacks
        minTokenLifetime = 5
        if not self.renewtokens:
            minTokenLifetime = -1
        
        self.callbacks=DeepJet_callbacks(self.keras_model,
                                    stop_patience=stop_patience, 
                                    lr_factor=lr_factor,
                                    lr_patience=lr_patience, 
                                    lr_epsilon=lr_epsilon, 
                                    lr_cooldown=lr_cooldown, 
                                    lr_minimum=lr_minimum,
                                    outputDir=self.outputDir,
                                    checkperiod=checkperiod,
                                    checkperiodoffset=self.trainedepoches,
                                    additional_plots=additional_plots,
                                    batch_loss = plot_batch_loss,
                                    minTokenLifetime = minTokenLifetime)
        
        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks=[additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)
        
        print('starting training')
        if load_in_mem:
            raise Exception('to be re-implemented later!')
        #else:
        
        #prepare generator 
        
        print("setting up generator... can take a while")
        self.train_data.invokeGenerator()
        self.val_data.invokeGenerator()
        #this is fixed
        nbatches_val = self.val_data.generator.getNBatches()
        nbatches_train = self.train_data.generator.getNBatches()
        #self.val_data.generator.debug=True
        #self.train_data.generator.debug=True
        #exit()
        
        while(self.trainedepoches < nepochs):
           
            #this can change from epoch to epoch
            print('>>>>Epoch', self.trainedepoches,"/",nepochs)
            print('training batches: ',nbatches_train)
            print('validation batches: ',nbatches_val)
            #calculate steps for this epoch
            #feed info below
            self.train_data.generator.prepareNextEpoch()
            self.val_data.generator.prepareNextEpoch()
                
            self.keras_model.fit_generator(self.train_data.generatorFunction() ,
                                           steps_per_epoch=nbatches_train, 
                                           epochs=self.trainedepoches + 1,
                                           initial_epoch=self.trainedepoches,
                                           callbacks=self.callbacks.callbacks,
                                           validation_data=self.val_data.generatorFunction(),
                                           validation_steps=nbatches_val, #)#,
                                           max_queue_size=1, #handled by DJC
                                           validation_freq=1,
                                           use_multiprocessing=False, #the threading one doe not loke DJC
                                           **trainargs)
            self.trainedepoches += 1
            self.train_data.generator.shuffleFilelist()
            nbatches_train = self.train_data.generator.getNBatches() #might have changed due to shuffeling
            #
        
        self.saveModel("KERAS_model.h5")
        del self.train_data.generator
        del self.val_data.generator
        return self.keras_model, self.callbacks.history
    
    
    
       
    def change_learning_rate(self, new_lr):
        import keras.backend as K
        if self.GAN_mode:
            K.set_value(self.discriminator.optimizer.lr, new_lr)
            K.set_value(self.gan.optimizer.lr, new_lr)
        else:
            K.set_value(self.keras_model.optimizer.lr, new_lr)
        
        
    
    
        
    
