


## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore import DataCollection
import tensorflow.keras as keras
import tensorflow as tf
import copy
from .gpuTools import DJCSetGPUs

from ..customObjects import get_custom_objects
custom_objects_list = get_custom_objects()


##helper





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
        
        scriptname=sys.argv[0]
        
        if parser is None: parser = ArgumentParser('Run the training')
        parser.add_argument('inputDataCollection')
        parser.add_argument('outputDir')
        parser.add_argument('--modelMethod', help='Method to be used to instantiate model in derived training class', metavar='OPT', default=None)
        parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
        parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
        parser.add_argument("--submitbatch",  help="submits the job to condor" , default=False, action="store_true")
        parser.add_argument("--walltime",  help="sets the wall time for the batch job, format: 1d5h or 2d or 3h etc" , default='1d')
        parser.add_argument("--isbatchrun",   help="is batch run", default=False, action="store_true")
        parser.add_argument("--valdata",   help="set validation dataset (optional)", default="")
        parser.add_argument("--takeweights",   help="Applies weights from the model given as relative or absolute path. Matches by names and skips layers that don't match.", default="")
        
        
        args = parser.parse_args()
        self.args = args
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
        DJCSetGPUs(args.gpu)
        
        if args.gpufraction>0 and args.gpufraction<1:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            from tensorflow.keras import backend as K
            K.set_session(sess)
            print('using gpu memory fraction: '+str(args.gpufraction))
        
        self.ngpus=1
        self.dist_strat_scope=None
        if len(args.gpu):
            self.ngpus=len([i for i in args.gpu.split(',')])
            print('running on '+str(self.ngpus)+ ' gpus')
            if self.ngpus > 1:
                self.dist_strat_scope = tf.distribute.MirroredStrategy()
            
        self.keras_inputs=[]
        self.keras_inputsshapes=[]
        self.keras_model=None
        self.keras_model_method=args.modelMethod
        self.keras_weight_model_path=args.takeweights
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
                var = input('output dir exists. To recover a training, please type "yes"\n')
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
            try:
                shutil.copyfile(scriptname,self.outputDir+os.path.basename(scriptname))
            except shutil.SameFileError:
                pass
            except BaseException as e:
                raise e
                
            self.copied_script = self.outputDir+os.path.basename(scriptname)
        else:
            self.copied_script = scriptname
        
        self.train_data = collection_class()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights=useweights
        
        if len(args.valdata):
            print('using validation data from ',args.valdata)
            self.val_data = DataCollection(args.valdata)
        
        else:
            if testrun:
                if len(self.train_data)>1:
                    self.train_data.split(testrun_fraction)
            
                self.train_data.dataclass_instance=None #can't be pickled
                self.val_data=copy.deepcopy(self.train_data)
                
            else:    
                self.val_data=self.train_data.split(splittrainandtest)
        


        shapes = self.train_data.getNumpyFeatureShapes()
        inputdtypes = self.train_data.getNumpyFeatureDTypes()
        inputnames= self.train_data.getNumpyFeatureArrayNames()
        for i in range(len(inputnames)): #in case they are not named
            if inputnames[i]=="" or inputnames[i]=="_rowsplits":
                inputnames[i]="input_"+str(i)+inputnames[i]


        print("shapes", shapes)
        print("inputdtypes", inputdtypes)
        print("inputnames", inputnames)
        
        self.keras_inputs=[]
        self.keras_inputsshapes=[]

        for s,dt,n in zip(shapes,inputdtypes,inputnames):
            self.keras_inputs.append(keras.layers.Input(shape=s, dtype=dt, name=n))
            self.keras_inputsshapes.append(s)
            
        #bookkeeping
        self.train_data.writeToFile(self.outputDir+'trainsamples.djcdc',abspath=True)
        self.val_data.writeToFile(self.outputDir+'valsamples.djcdc',abspath=True)
            
        if not isNewTraining:
            kfile = self.outputDir+'/KERAS_check_model_last.h5'
            if not os.path.isfile(kfile):
                kfile = self.outputDir+'/KERAS_check_model_last' #savedmodel format
                if not os.path.isdir(kfile):
                    kfile=''
            if len(kfile):
                print('loading model',kfile)
                
                if self.dist_strat_scope is not None:
                    with self.dist_strat_scope.scope():
                        self.loadModel(kfile)
                else:
                    self.loadModel(kfile)
                self.trainedepoches=0
                if os.path.isfile(self.outputDir+'losses.log'):
                    for line in open(self.outputDir+'losses.log'):
                        valloss = line.split(' ')[1][:-1]
                        if not valloss == "None":
                            self.trainedepoches+=1
                else:
                    print('incomplete epochs, starting from the beginning but with pretrained model')
            else:
                print('no model found in existing output dir, starting training from scratch')
        
    def __del__(self):
        if hasattr(self, 'train_data'):
            del self.train_data
            del self.val_data
        
    def modelSet(self):
        return (not self.keras_model==None) and not len(self.keras_weight_model_path)
        
    def setDJCKerasModel(self,model,*args,**kwargs): 
        if len(self.keras_inputs)<1:
            raise Exception('setup data first')   
        self.keras_model=model(*args,**kwargs)
        if hasattr(self.keras_model, "_is_djc_keras_model"):
            self.keras_model.setInputShape(self.keras_inputs)
            self.keras_model.build(None)
        if not self.keras_model:
            raise Exception('Setting DJCKerasModel not successful') 
        
        
    def setModel(self,model,**modelargs):
        if len(self.keras_inputs)<1:
            raise Exception('setup data first') 
        if self.dist_strat_scope is not None:
            with self.dist_strat_scope.scope():
                self.keras_model=model(self.keras_inputs,**modelargs)
        else:
            self.keras_model=model(self.keras_inputs,**modelargs)
        if hasattr(self.keras_model, "_is_djc_keras_model"): #compatibility
            self.keras_model.setInputShape(self.keras_inputs)
            self.keras_model.build(None)
            
        if len(self.keras_weight_model_path):
            from DeepJetCore.modeltools import apply_weights_where_possible, load_model
            self.keras_model = apply_weights_where_possible(self.keras_model, 
                                         load_model(self.keras_weight_model_path))
        #try:
        #    self.keras_model=model(self.keras_inputs,**modelargs)
        #except BaseException as e:
        #    print('problem in setting model. Reminder: since DJC 2.0, NClassificationTargets and RegressionTargets must not be specified anymore')
        #    raise e
        if not self.keras_model:
            raise Exception('Setting model not successful') 
        
    
    def saveCheckPoint(self,addstring=''):
        
        self.checkpointcounter=self.checkpointcounter+1 
        self.saveModel("KERAS_model_checkpoint_"+str(self.checkpointcounter)+"_"+addstring)    
           
    
    def _loadModel(self,filename):
        from tensorflow.keras.models import load_model
        keras_model=load_model(filename, custom_objects=custom_objects_list)
        optimizer=keras_model.optimizer
        return keras_model, optimizer
                
    def loadModel(self,filename):
        self.keras_model, self.optimizer = self._loadModel(filename)
        self.compiled=True
        if self.ngpus>1:
            self.compiled=False
        
    def setCustomOptimizer(self,optimizer):
        self.optimizer = optimizer
        self.custom_optimizer=True
        
    def compileModel(self,
                     learningrate,
                     clipnorm=None,
                     print_models=False,
                     metrics=None,
                     is_eager=False,
                     **compileargs):
        if not self.keras_model and not self.GAN_mode:
            raise Exception('set model first') 

        if self.ngpus>1 and not self.submitbatch:
            print('Model being compiled for '+str(self.ngpus)+' gpus')
            
        self.startlearningrate=learningrate
        
        if not self.custom_optimizer:
            from tensorflow.keras.optimizers import Adam
            if clipnorm:
                self.optimizer = Adam(lr=self.startlearningrate,clipnorm=clipnorm)
            else:
                self.optimizer = Adam(lr=self.startlearningrate)
            
            
        
        if self.dist_strat_scope is not None:
            with self.dist_strat_scope.scope():
                self.keras_model.compile(optimizer=self.optimizer,metrics=metrics,**compileargs)
        else:
            self.keras_model.compile(optimizer=self.optimizer,metrics=metrics,**compileargs)
            
        if is_eager:
            #call on one batch to fully build it
            self.keras_model(self.train_data.getExampleFeatureBatch())
            
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
        


        #import h5py
        #f = h5py.File(self.outputDir+outfile, 'r+')
        #del f['optimizer_weights']
        #f.close()
        
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
        
        
        #make sure tokens don't expire
        from .tokenTools import checkTokens, renew_token_process
        from _thread import start_new_thread
        
        if self.renewtokens:
            print('afs backgrounder has proven to be unreliable, use with care')
            checkTokens()
            start_new_thread(renew_token_process,())
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        
        
    def trainModel(self,
                   nepochs,
                   batchsize,
                   run_eagerly=False,
                   batchsize_use_sum_of_squares = False,
                   fake_truth=False,#extend the truth list with dummies. Useful when adding more prediction outputs than truth inputs
                   stop_patience=-1, 
                   lr_factor=0.5,
                   lr_patience=-1, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   checkperiod=10,
                   backup_after_batches=-1,
                   additional_plots=None,
                   additional_callbacks=None,
                   load_in_mem = False,
                   max_files = -1,
                   plot_batch_loss = False,
                   **trainargs):
        
        
        self.keras_model.run_eagerly=run_eagerly
        # write only after the output classes have been added
        self._initTraining(nepochs,batchsize, batchsize_use_sum_of_squares)
        
        try: #won't work for purely eager models
            self.keras_model.save(self.outputDir+'KERAS_untrained_model')
        except:
            pass
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
                                    backup_after_batches=backup_after_batches,
                                    checkperiodoffset=self.trainedepoches,
                                    additional_plots=additional_plots,
                                    batch_loss = plot_batch_loss,
                                    print_summary_after_first_batch=run_eagerly,
                                    minTokenLifetime = minTokenLifetime)
        
        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks=[additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)
            
        
        print('starting training')
        if load_in_mem:
            print('make features')
            X_train = self.train_data.getAllFeatures(nfiles=max_files)
            X_test = self.val_data.getAllFeatures(nfiles=max_files)
            print('make truth')
            Y_train = self.train_data.getAllLabels(nfiles=max_files)
            Y_test = self.val_data.getAllLabels(nfiles=max_files)
            self.keras_model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepochs,
                                 callbacks=self.callbacks.callbacks,
                                 validation_data=(X_test, Y_test),
                                 max_queue_size=1,
                                 use_multiprocessing=False,
                                 workers=0,    
                                 **trainargs)
        else:
        
            #prepare generator 
        
            print("setting up generator... can take a while")
            use_fake_truth=None
            if fake_truth:
                if isinstance(self.keras_model.output,dict):
                    use_fake_truth = [k for k in self.keras_model.output.keys()]
                elif isinstance(self.keras_model.output,list):
                    use_fake_truth = len(self.keras_model.output)
                    
            traingen = self.train_data.invokeGenerator(fake_truth = use_fake_truth)
            valgen = self.val_data.invokeGenerator(fake_truth = use_fake_truth)


            while(self.trainedepoches < nepochs):
           
                #this can change from epoch to epoch
                #calculate steps for this epoch
                #feed info below
                traingen.prepareNextEpoch()
                valgen.prepareNextEpoch()
                nbatches_train = traingen.getNBatches() #might have changed due to shuffeling
                nbatches_val = valgen.getNBatches()
            
                print('>>>> epoch', self.trainedepoches,"/",nepochs)
                print('training batches: ',nbatches_train)
                print('validation batches: ',nbatches_val)
                
                self.keras_model.fit(traingen.feedNumpyData(), 
                                     steps_per_epoch=nbatches_train,
                                     epochs=self.trainedepoches + 1,
                                     initial_epoch=self.trainedepoches,
                                     callbacks=self.callbacks.callbacks,
                                     validation_data=valgen.feedNumpyData(),
                                     validation_steps=nbatches_val,
                                     max_queue_size=1,
                                     use_multiprocessing=False,
                                     workers=0,
                                     **trainargs
                )
                self.trainedepoches += 1
                traingen.shuffleFileList()
                #
        
            self.saveModel("KERAS_model.h5")

        return self.keras_model, self.callbacks.history
    
    
    
       
    def change_learning_rate(self, new_lr):
        import tensorflow.keras.backend as K
        if self.GAN_mode:
            K.set_value(self.discriminator.optimizer.lr, new_lr)
            K.set_value(self.gan.optimizer.lr, new_lr)
        else:
            K.set_value(self.keras_model.optimizer.lr, new_lr)
        
        
    
    
        
    
