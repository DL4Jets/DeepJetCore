from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D
from keras.models import Model
from pdb import set_trace
from Layers import GradientReversal

def dense_model(Inputs,nclasses,nregressions,dropoutRate=0.25):
    """
    Dense matrix, defaults similat to 2016 training
    """
    if nregressions: raise ValueError('The dense model does not support regression, only classification')
    #set_trace()
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def dense_model_moments(Inputs,nclasses,nregressions,dropoutRate=0.25):
	"""
	Dense matrix, defaults similat to 2016 training. Adaptation for moment-based domain adaptation
	"""
	if nregressions: raise ValueError(
			'The dense model does not support regression, only classification')
	#set_trace()
	#  Here add e.g. the normal dense stuff from DeepCSV
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
	x = Dropout(dropoutRate)(x)
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	x = Dropout(dropoutRate)(x)
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	x = Dropout(dropoutRate)(x)
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	x = Dropout(dropoutRate)(x)
	x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
	model = Model(inputs=Inputs, outputs=[predictions, predictions])
	return model

def dense_model_gradientReversal(Inputs,nclasses,nregressions,dropoutRate=0.25):
	"""
	Dense matrix, defaults similat to 2016 training. Adaptation for gradient reversal
	domain adaptation
	"""
	if nregressions: raise ValueError(
			'The dense model does not support regression, only classification')
	#set_trace()
	#  Here add e.g. the normal dense stuff from DeepCSV
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
	x = Dropout(dropoutRate)(x)
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	x = Dropout(dropoutRate)(x)
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	x_mid = Dropout(dropoutRate)(x)
	x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x_mid)
	x = Dropout(dropoutRate)(x)
	x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
	predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
	
	dom_ada = GradientReversal()(x_mid)
	dom_ada = Dense(100, activation='relu', kernel_initializer='lecun_uniform')(dom_ada)
	dom_ada = Dropout(dropoutRate)(dom_ada)
	dom_ada = Dense(100, activation='relu', kernel_initializer='lecun_uniform')(dom_ada)
	dom_ada = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(dom_ada)
	
	model = Model(inputs=Inputs, outputs=[predictions, dom_ada])
	return model


def dense_model_reg_fake(Inputs,nclasses,Inputshape,dropoutRate=0.25):
   """ 
   Somewhat of a fake to test how much the BTV variables helped, only give REC PT and genPT. BTV and recvo do not get merged! You need to set BTV loss to weight 0!
   """
   x = Dense(1, activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
 
   # only this is really used
   predictflav=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='flavour_pred')(x)
   
   flavpt=Inputs[1]
   flavpt=Dense(10, activation='relu',kernel_initializer='lecun_uniform')(flavpt)
 
   predictions = [predictflav,
                  Dense(1, activation='linear',kernel_initializer='normal',name='pt_pred')(flavpt)]
   model = Model(inputs=Inputs, outputs=predictions)
   return model


def dense_model_reg(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    Dense matrix, defaults similar to 2016 training now with regression
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    #x =  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    #x =  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    
    predictflav=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='flavour_pred')(x)
   
    flavpt=merge( [x,Inputs[1]] , mode='concat')
    flavpt=Dense(10, activation='relu',kernel_initializer='lecun_uniform')(flavpt)
    #flavpt=Dense(10, activation='linear',kernel_initializer='lecun_uniform')(flavpt)
   
    predictions = [predictflav,
                   Dense(1, activation='linear',kernel_initializer='normal',name='pt_pred')(flavpt)]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def dense_model2(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    Dense matrix, defaults similat to 2016 training
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform',input_shape=Inputshape)(Inputs)
    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def dense_model_broad_flat(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    cpf = Flatten()(Inputs[1])
    npf = Flatten()(Inputs[2])
    vtx = Flatten()(Inputs[3])
    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    
    x=  Dense(600, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(600, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(300, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(300, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def dense_model_microPF(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    from keras.layers.local import LocallyConnected1D   
    npf = Flatten()(Inputs[1])
    
    x = merge( [Inputs[0],npf] , mode='concat')
    
    x=  Dense(250, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model
