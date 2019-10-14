


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
import keras
if float(keras.__version__[2:5]) >= 2.2:
    from keras.utils import multi_gpu_model
else:
    def multi_gpu_model(m, ngpus):
        return m

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
        if float(keras.__version__[2:]) < 2.2:
            print('multi gpu option from keras >= 2.2.2 is NOT available for now. (see DJC issues 28 and 30)')
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
				renewtokens=True,
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
            except ImportError:
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
        


        shapes=self.train_data.getInputShapes()
        self.train_data.maxFilesOpen=-1
        
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
        self.keras_model=model(self.keras_inputs,
                               self.train_data.getNClassificationTargets(),
                               self.train_data.getNRegressionTargets(),
                               **modelargs)
        if not self.keras_model:
            raise Exception('Setting model not successful') 
        
    def setGANModel(self, generator_func, discriminator_func):
        '''
        The inputs are functions that generate a keras Model 
        The GAN output must match the discriminator input.
        The first and only function argument of the discriminator must be the input.
        The generator MUST get the same input. (e.g. both get a list of which one item is the
        discriminator input, the other the generator input)
        '''  
        self.GAN_mode = True
        self.create_generator     = generator_func
        self.create_discriminator = discriminator_func
        
    def _create_gan(self,discriminator, generator, gan_input):
        import keras
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = keras.models.Model(inputs=gan_input, outputs=gan_output)
        return gan
        
    def defineCustomPredictionLabels(self, labels):
        self.train_data.defineCustomPredictionLabels(labels)
        self.val_data.defineCustomPredictionLabels(labels)
    
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
                     generator_loss=None,
                     print_models=False,
                     discr_loss_weights=None,
                     gan_loss_weights=None,
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
            
            
        if self.GAN_mode:
            if metrics is None:
                metrics=['accuracy']
            else:
                if not ('accuracy' in metrics):
                    metrics = ['accuracy']+metrics
                    
            self.generator= self.create_generator(self.keras_inputs)
            if generator_loss is None:
                generator_loss = [null_loss for i in range(len(self.generator.outputs))]
            self.generator.compile(optimizer=self.optimizer,loss=generator_loss,metrics=metrics,**compileargs)
            
            self.discriminator= self.create_discriminator(self.keras_inputs)
            self.discriminator.compile(optimizer=self.optimizer,loss=discriminator_loss,loss_weights=discr_loss_weights,metrics=metrics,**compileargs)
            
            self.discriminator.trainable=False
            self.gan = self._create_gan(self.discriminator, self.generator, self.keras_inputs)
            self.gan.compile(optimizer=self.optimizer,loss=discriminator_loss,loss_weights=gan_loss_weights,metrics=metrics,**compileargs)
            
            if print_models:
                print('GENERATOR:')
                print(self.generator.summary())
                print('DISCRIMINATOR:')
                print(self.discriminator.summary())
                print('GAN:')
                print(self.gan.summary())
        else:    
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
        import tensorflow as tf
        import keras.backend as K
        tfsession=K.get_session()
        saver = tf.train.Saver()
        tfoutpath=self.outputDir+outfile+'_tfsession/tf'
        import os
        os.system('rm -rf '+tfoutpath)
        os.system('mkdir -p '+tfoutpath)
        saver.save(tfsession, tfoutpath)


        #import h5py
        #f = h5py.File(self.outputDir+outfile, 'r+')
        #del f['optimizer_weights']
        #f.close()
        
    def _initTraining(self,
                      nepochs,
                     batchsize,maxqsize):
        
        
        if self.submitbatch:
            from DeepJetCore.training.batchTools import submit_batch
            submit_batch(self, self.args.walltime)
            exit() #don't delete this!
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.writeToFile(self.outputDir+'trainsamples.dc')
        self.val_data.writeToFile(self.outputDir+'valsamples.dc')
        
        #make sure tokens don't expire
        from .tokenTools import checkTokens, renew_token_process
        from thread import start_new_thread
        
        if self.renewtokens:
            print('starting afs backgrounder')
            checkTokens()
            start_new_thread(renew_token_process,())
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        
        averagesamplesperfile=self.train_data.getAvEntriesPerFile()
        samplespreread=maxqsize*batchsize
        nfilespre=max(int(samplespreread/averagesamplesperfile),2)
        nfilespre+=1
        nfilespre=min(nfilespre, len(self.train_data.samples)-1)
        #if nfilespre>15:nfilespre=15
        print('best pre read: '+str(nfilespre)+'  a: '+str(int(averagesamplesperfile)))
        print('total sample size: '+str(self.train_data.nsamples))
        #exit()
        
        if self.train_data.maxFilesOpen<0 or True:
            self.train_data.maxFilesOpen=nfilespre
            self.val_data.maxFilesOpen=min(int(nfilespre/2),1)
        
    def trainModel(self,
                   nepochs,
                   batchsize,
                   stop_patience=-1, 
                   lr_factor=0.5,
                   lr_patience=-1, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   maxqsize=5, 
                   checkperiod=10,
                   additional_plots=None,
                   additional_callbacks=None,
                   load_in_mem = False,
                   plot_batch_loss = False,
                   **trainargs):
        
        
        # check a few things, e.g. output dimensions etc.
        # need implementation, but probably TF update SWAPNEEL
        customtarget=self.train_data.getCustomPredictionLabels()
        if customtarget:
            pass
            # work on self.model.outputs
            # check here if the output dimension of the model fits the custom labels
        
        # write only after the output classes have been added
        self._initTraining(nepochs,batchsize,maxqsize)
        
        #self.keras_model.save(self.outputDir+'KERAS_check_last_model.h5')
        print('setting up callbacks')
        from .DeepJet_callbacks import DeepJet_callbacks
        
        
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
                                    batch_loss = plot_batch_loss)
        
        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks=[additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)
        
        print('starting training')
        if load_in_mem:
            print('make features')
            X_train = self.train_data.getAllFeatures()
            X_test = self.val_data.getAllFeatures()
            print('make truth')
            Y_train = self.train_data.getAllLabels()
            Y_test = self.val_data.getAllLabels()
            self.keras_model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepochs,
                                 callbacks=self.callbacks.callbacks,
                                 validation_data=(X_test, Y_test),
                                 **trainargs)
        else:
            self.keras_model.fit_generator(self.train_data.generator() ,
                                           steps_per_epoch=self.train_data.getNBatchesPerEpoch(), 
                                           epochs=nepochs-self.trainedepoches,
                                           callbacks=self.callbacks.callbacks,
                                           validation_data=self.val_data.generator(),
                                           validation_steps=self.val_data.getNBatchesPerEpoch(), #)#,
                                           max_queue_size=1,
                                           #max_q_size=1,
                                           use_multiprocessing=True, #the threading one doe not loke DJC
                                           **trainargs)
        
        self.trainedepoches=nepochs
        self.saveModel("KERAS_model.h5")
        
        import copy
        #reset all file reads etc
        tmpdc=copy.deepcopy(self.train_data)
        del self.train_data
        self.train_data=tmpdc
        
        return self.keras_model, self.callbacks.history
    
    
    
    def trainGAN_exp(self,
                     nepochs,
                     batchsize,
                 gan_skipping_factor=1,
                 discr_skipping_factor=1,
                     verbose=1,
                     checkperiod=1,
                     additional_plots=None,
                   additional_callbacks=None):
        
        self._initTraining(nepochs,batchsize,maxqsize=5)
        
        print('setting up callbacks')
        from .DeepJet_callbacks import DeepJet_callbacks
        
        #callbacks are just a placeholder for now
        self.callbacks=DeepJet_callbacks(self.keras_model,
                                    stop_patience=-1, 
                                    lr_factor=.9,
                                    lr_patience=-1, 
                                    lr_epsilon=1, 
                                    lr_cooldown=1, 
                                    lr_minimum=1,
                                    outputDir=self.outputDir,
                                    checkperiod=checkperiod,
                                    checkperiodoffset=self.trainedepoches,
                                    additional_plots=additional_plots)
        self.callbacks.callbacks=[]
        #needs more dedicated callbacks
        
        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks=[additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)
            
        
        
        gan_history, _ = self.gan_fit_generator(generator=self.train_data.generator(),
                               datacollection=self.train_data,
                               steps_per_epoch=self.train_data.getNBatchesPerEpoch(),
                               epochs=nepochs,
                               verbose=verbose,
                               callbacks_discriminator=self.callbacks.callbacks,
                               callbacks_gan=None,
                               validation_data=self.val_data.generator(),
                               validation_steps=self.val_data.getNBatchesPerEpoch(),
                               validation_freq=1,
                               class_weight=None,
                               gan_skipping_factor=gan_skipping_factor,
                               discr_skipping_factor=discr_skipping_factor,
                               max_queue_size=10,
                               initial_epoch=0)
        
        self.saveModel("KERAS_model.h5")
        return self.gan, gan_history
    
        
    def change_learning_rate(self, new_lr):
        import keras.backend as K
        if self.GAN_mode:
            K.set_value(self.discriminator.optimizer.lr, new_lr)
            K.set_value(self.gan.optimizer.lr, new_lr)
        else:
            K.set_value(self.keras_model.optimizer.lr, new_lr)
        
        
    
    def gan_fit_generator(self,
                      generator,
                      datacollection,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks_discriminator=None,
                      callbacks_gan=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      gan_skipping_factor=1,
                      discr_skipping_factor=1,
                      validation_freq=1,###TBI FIXME
                      max_queue_size=10,
                      initial_epoch=0,
                      recover_discriminator=True
                      ):
        """See docstring for `Model.fit_generator`."""
        
        import keras
        from sklearn.utils import shuffle
        import keras.callbacks as cbks
        #from keras.training_utils import should_run_validation
        from keras.utils.generic_utils import to_list
        import numpy as np
        
        epoch = initial_epoch
        
        do_validation = bool(validation_data)
        #DEBUG self.discriminator._make_train_function()
        #DEBUG self.gan._make_train_function()
        if do_validation and False: #DEBUG
            self.discriminator._make_test_function()
            self.gan._make_test_function()
        
        
        d_out_labels = ['dis_' + n for n in self.discriminator.metrics_names ]
        g_out_labels = ['gan_' + n for n in self.gan.metrics_names ]
        
        d_callback_metrics = d_out_labels + ['val_' + n for n in d_out_labels]
        g_callback_metrics = g_out_labels + ['val_' + n for n in g_out_labels]
        
        # prepare callbacks
        self.discriminator.history = cbks.History()
        self.gan.history = cbks.History()
        _callbacks = [cbks.BaseLogger(
            stateful_metrics=self.discriminator.stateful_metric_names)]
        _callbacks += [cbks.BaseLogger(
            stateful_metrics=self.gan.stateful_metric_names)]
        
        if verbose:
            _callbacks.append(
                cbks.ProgbarLogger(
                    count_mode='steps',
                    stateful_metrics=self.gan.stateful_metric_names)) #one model is enough here!#use only gan here
            
        callbacks_gan = callbacks_gan or []
        callbacks_discriminator = callbacks_discriminator or []
        for c in callbacks_gan:
            c.set_model(self.gan)
        for c in callbacks_discriminator:
            c.set_model(self.discriminator)
        
        _callbacks += (callbacks_gan) + (callbacks_discriminator) + [self.discriminator.history] + [self.gan.history]
        callbacks = cbks.CallbackList(_callbacks)
        
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': d_callback_metrics + g_callback_metrics,
        })
        #newer keras callbacks._call_begin_hook('train')
        callbacks.on_train_begin()
        
        enqueuer = None
        val_enqueuer = None
        
        try:
            if do_validation:
                
                val_data = validation_data
                val_enqueuer_gen = val_data
                              
                output_generator = generator
        
            ## callbacks.model.stop_training = False ##FIXME TBI
            # Construct epoch logs.
            epoch_logs = {}
            skip_gan_training = False
            while epoch < epochs:
                for m in self.discriminator.stateful_metric_functions:
                    m.reset_states()
                for m in self.gan.stateful_metric_functions:
                    m.reset_states()
                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0
                while steps_done < steps_per_epoch:
                    generator_output = next(output_generator)
        
                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
        
                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    if x is None or len(x) == 0:
                        # Handle data tensors support when no input given
                        # step-size = 1 for data tensors
                        batch_size = 1
                    elif isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    # build batch logs
                    batch_logs = {'batch': batch_index, 'size': batch_size}
                    callbacks.on_batch_begin(batch_index, batch_logs)
        
        
                    #GAN training here 
                    
                    x_gen = self.generator.predict(x)
                    
                    #DEBUG - NEEDS CALLBACK
                    # REMOVE IN FULL VERSION
                    if False and steps_done%50:
                        forplots = np.concatenate([x_gen[0][:4], x[0][:4]],axis=0)
                        from tools import quickplot, plotgrid
                        plotgrid(forplots, nplotsx=4, nplotsy=2, outname="merged.pdf")
                        quickplot(x_gen[0][0], "gen.pdf")
                        quickplot(x[0][0], "data.pdf")
                    
                    #this needs to be more generic and actually done for every list item
                    #replaceTruthForGAN gives a list
                    
                    adapted_truth_data = datacollection.replaceTruthForGAN(
                                                  generated_array=np.zeros(batch_size, dtype='float32')+1, 
                                                  original_truth=y)
                    
                    adapted_truth_generated = datacollection.replaceTruthForGAN(
                                                  generated_array=np.zeros(batch_size, dtype='float32'), 
                                                  original_truth=y)
                    
                    y_dis = [np.concatenate([adapted_truth_data[i],adapted_truth_generated[i]],axis=0) \
                             for i in range(len(adapted_truth_data))]
                    
                    x_dis = [np.concatenate([x[i],x_gen[i]],axis=0) 
                                 for i in range(len(x))]
                    
                    y_dis_new =  [shuffle(n, random_state=steps_done) for n in y_dis]
                    x_dis_new =  [shuffle(n, random_state=steps_done) for n in x_dis]
                    
                    y_dis_b1 = [y_dis_new[i][:batch_size,...] for i in range(len(y_dis_new))]
                    y_dis_b2 = [y_dis_new[i][batch_size:,...] for i in range(len(y_dis_new))]
                    
                    x_dis_b1 = [x_dis_new[i][:batch_size,...] for i in range(len(x_dis_new))]
                    x_dis_b2 = [x_dis_new[i][batch_size:,...] for i in range(len(x_dis_new))]
                    
                    #add [:batch_size,...]
                    #to the above for cut-off
                    # TBI TBI FIXME
                    
                    ## FIXME: cut in half to have same batch size everywhere
                    # also here would be the place to implement weighting of discr versus gen
                    
                    if (not batch_index%discr_skipping_factor):
                        self.discriminator.trainable=True
                        outs = self.discriminator.train_on_batch(x_dis_b1, y_dis_b1,
                                                    sample_weight=sample_weight,
                                                    class_weight=class_weight)
                        
                        outs = self.discriminator.train_on_batch(x_dis_b2, y_dis_b2,
                                                    sample_weight=sample_weight,
                                                    class_weight=class_weight)
                        
                        outs = to_list(outs)
                        
                        if recover_discriminator:
                            if outs[1] < 0.5:
                                skip_gan_training=True
                            else:
                                skip_gan_training=False
                        for l, o in zip(d_out_labels, outs):
                            batch_logs[l] = o
                            
                    
                    if (not skip_gan_training) and (not batch_index%gan_skipping_factor):
                        self.discriminator.trainable=False
                        y_gen = np.zeros(batch_size, dtype='float32')+1.
                        outs = self.gan.train_on_batch(x, y_gen,
                                                    sample_weight=sample_weight,
                                                    class_weight=class_weight)
                        outs = to_list(outs)
                        for l, o in zip(g_out_labels, outs):
                            batch_logs[l] = o    
        
                    #callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)
                    callbacks.on_batch_end(batch_index, batch_logs)
        
                    batch_index += 1
                    steps_done += 1
        
                    # Epoch finished.
                    if (steps_done >= steps_per_epoch and
                            do_validation):
                        # Note that `callbacks` here is an instance of
                        # `keras.callbacks.CallbackList`
                        
                        ## this evaluate will get problems with the truth definition
                        ## needs to be fixed in the generator? Or just make traindata do it?
                        
                        val_outs = self.discriminator.evaluate_generator(
                                val_enqueuer_gen,
                                validation_steps,
                                #callbacks=callbacks,
                                workers=0)
                        
                        val_outs = to_list(val_outs)
                        # Same labels assumed.
                        for l, o in zip(d_out_labels, val_outs):
                            epoch_logs['val_' + l] = o
                            
                        val_outs = self.gan.evaluate_generator(
                                val_enqueuer_gen,
                                validation_steps,
                                #callbacks=callbacks,
                                workers=0)
                        
                        val_outs = to_list(val_outs)
                        # Same labels assumed.
                        for l, o in zip(g_out_labels, val_outs):
                            epoch_logs['val_' + l] = o
        
                    #if callbacks.model.stop_training:  ##FIXME TBI
                    #    break
        
                callbacks.on_epoch_end(epoch, epoch_logs)
                epoch += 1
                #if callbacks.model.stop_training:  ##FIXME TBI
                #    break
        
        finally:
            try:
                if enqueuer is not None:
                    enqueuer.stop()
            finally:
                if val_enqueuer is not None:
                    val_enqueuer.stop()
        
        #callbacks._call_end_hook('train')
        callbacks.on_train_end()
        return self.gan.history , self.discriminator.history

        
    
