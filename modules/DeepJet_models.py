#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D
#fix for dropout on gpus

#import tensorflow
#from tensorflow.python.ops import control_flow_ops 
#tensorflow.python.control_flow_ops = control_flow_ops


def Model_FatJet(Inputs,nclasses,dropoutRate=-1):

    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf = Flatten()(cpf)
    
    cmap = Conv2D(4, 3, 3, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    cmap =  Dropout(dropoutRate)(cmap)
    cmap = Flatten()(cmap)
        
    #merge with the flobals
    x = merge( [Inputs[0],cpf, cmap] , mode='concat')
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(x) 
    
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    

def RecurrenPT(Inputs,nclasses,dropoutRate=-1):

    x_pt = Masking()(Inputs[1])
    x_pt = LSTM(100)(x_pt)
    x = merge( [x_pt, Inputs[0]] , mode='concat')
    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = merge( [x, Inputs[2]] , mode='concat')
    predictions = [Dense(2, activation='linear',init='normal')(x),Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def Schwartz_gluon_model(Inputs,nclasses,dropoutRate=-1):
     x =   Convolution2D(64, (8,8)  , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(Inputs[1])
     x = MaxPooling2D(pool_size=(2, 2))(x)
     x =   Convolution2D(64, (4,4) , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
     x = MaxPooling2D(pool_size=(2, 2))(x)
     x =   Convolution2D(64, (4,4)  , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
     x = MaxPooling2D(pool_size=(2, 2))(x)
     x = Flatten()(x)
     x = merge( [x, Inputs[1]] , mode='concat')
    # linear activation for regression and softmax for classification
     x = Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x)     
     predictions = [Dense(2, activation='linear',init='normal')(x),Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)]
     model = Model(inputs=Inputs, outputs=predictions)
     return model

def Serious_gluon_model(Inputs,nclasses,dropoutRate=-1):
     x =   LocallyConnected2D(64, (8,8) ,stride= (4,4) , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(Inputs[1])
#     x = MaxPooling2D(pool_size=(2, 2))(x)
     x = Convolution2D(64, (4,4) , 1 , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
     x = Convolution2D(64, (4,4) , 1 , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
     x = MaxPooling2D(pool_size=(2, 2))(x)
     x = Flatten()(x)
     x = merge( [x, Inputs[0]] , mode='concat')
    # linear activation for regression and softmax for classification
     x = Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x) 
     x = Dense(128, activation='relu',kernel_initializer='lecun_uniform')(x) 
     x = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(x) 
     x = Dense(64, activation='relu',kernel_initializer='lecun_uniform')(x) 
     
     predictions = [Dense(2, activation='linear',init='normal')(x),Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)]
     model = Model(inputs=Inputs, outputs=predictions)
     return model


def Incept_model(Inputs,dropoutRate=0.25):
    """
        This NN adds two inputs, one for a conv net and a seceond for a dense net, both nets get combined. The last layer is split into regression and classification activations (softmax, linear)
    """
    
    x =   Convolution2D(20, 1 , 1, border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
    # add more layers to get deeper
    x =   Convolution2D(10, 1 , 1, border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x =   Convolution2D(5, 1 , 1, border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Flatten()(x)
    #  Here add e.g. the normal dense stuff from DeepCSV
    y = Dense(1, activation='relu',kernel_initializer='lecun_uniform',input_shape=(1,))(Inputs[1])
    y = Dense(1, activation='relu',kernel_initializer='lecun_uniform')(y)
    y = Dense(1, activation='relu',kernel_initializer='lecun_uniform')(y)
    # add more layers to get deeper
    
    # combine convolutional and dense (global) layers
    x = merge( [x,y ] , mode='concat')
    
    # linear activation for regression and softmax for classification
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    # add more layers to get deeper
    
    predictions = [Dense(1, activation='linear',init='normal')(x),Dense(5, activation='softmax',kernel_initializer='lecun_uniform')(x)]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def Dense_model(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    Dense matrix, defaults similat to 2016 training
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform',input_shape=Inputshape)(Inputs)
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

def Dense_model_reg_fake(Inputs,nclasses,Inputshape,dropoutRate=0.25):
   """ 
   Somewhat of a fake to test how much the BTV variables helped, only give REC PT and genPT. BTV and reco do not get merged! You need to set BTV loss to weight 0!
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


def Dense_model_reg(Inputs,nclasses,Inputshape,dropoutRate=0.25):
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

def Dense_model2(Inputs,nclasses,Inputshape,dropoutRate=0.25):
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



def Dense_model_broad_flat(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    
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
   
from keras import backend as K
from keras.layers.core import Lambda 

def mult_zeros(a):
    #zeros = K.reshape(a[1],(a[1].shape[0],a[1].shape[1],1))
    a0 = a[0]*a[1]
    return a0

def Dense_model_broad_rec_zeros(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour' using recurrent layers with zero masking, very very slow due to masking?
    """  
    
    UseConv=True

    cpf=Inputs[1]
    if UseConv:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    
        zeros_cpf = Inputs[4]
        # 
        # in case only one zero per vector
        #    zeros_cpf = Reshape((25,1))(zeros_cpf)
        #    cpf = Lambda(mult_zeros)([cpf,zeros_cpf])
        # in case of vectors of os.
        cpf = merge([cpf, Permute((2,1))(zeros_cpf)],mode='mul')
        cpf = Masking()(cpf)

    cpf  = LSTM(150,go_backwards=True)(cpf)
     
    #cpf  = LSTM(50,go_backwards=True)(cpf)
#    cpf = Flatten()(cpf)
    

    npf=Inputs[2]
    if UseConv:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        print ('sahe', ' ' , npf.shape)
    #npf = Flatten()(npf)

    zeros_npf = Inputs[5]
    # 
    # in case only one zero per vector
#    zeros_npf = Reshape((25,1))(zeros_npf)
 #   npf = Lambda(mult_zeros)([npf,zeros_npf])

    # in case of vectors of os.
#    npf = merge([npf,Permute((2,1))(zeros_npf)],mode='mul')

    npf = Masking()(npf)
    npf = LSTM(50,go_backwards=True)(npf)

    vtx = Inputs[3]
    if UseConv:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)


   
#    vtx = Flatten()(vtx)
    zeros_vtx = Inputs[6]
    # 
    # in case only one zero per vector
#    zeros_vtx = Reshape((4,1))(zeros_vtx)
#    vtx = Lambda(mult_zeros)([npf,zeros_vtx])
    # in case of vectors of os.
    vtx = merge([vtx,Permute((2,1))(zeros_vtx)],mode='mul')
    vtx = Masking()(vtx)
    vtx = LSTM(50,go_backwards=True,unroll=True)(vtx)

    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    

    x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def Dense_model_broad_rec(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour' using recurrent layers with zero masking, very very slow due to masking?
    """  
    
    UseConv=True

    cpf=Inputs[1]
    if UseConv:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    
    cpf  = LSTM(150,go_backwards=True)(cpf)
    npf=Inputs[2]
    if UseConv:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)

    npf = LSTM(50,go_backwards=True)(npf)

    vtx = Inputs[3]
    if UseConv:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)


    vtx = LSTM(50,go_backwards=True,unroll=True)(vtx)

    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    

    x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def Dense_model_broad(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    

    UseConv=True
    cpf=Inputs[1]
    if UseConv:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        
    cpf = Flatten()(cpf)
    
    npf=Inputs[2]
    if UseConv:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        
    npf = Flatten()(npf)
    
    vtx = Inputs[3]
    if UseConv:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        
    vtx = Flatten()(vtx)
    
    
    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    

    x=  Dense(350, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def Dense_model_broad_map(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    

    UseConv=True
    cpf=Inputs[1]
    if UseConv:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        
    cpf = Flatten()(cpf)
    
    npf=Inputs[2]
    if UseConv:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        
    npf = Flatten()(npf)
    
    vtx = Inputs[3]
    if UseConv:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        
    vtx = Flatten()(vtx)
    
    
    cmap = Conv2D(4, 3, 3, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[4])
    cmap =  Dropout(dropoutRate)(cmap)
    cmap=Flatten()(cmap)
    nmap = Conv2D(2, 3, 3, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[5])
    nmap = Dropout(dropoutRate)(nmap)
    nmap=Flatten()(nmap)
    
    x = merge( [Inputs[0],cpf,npf,vtx, cmap, nmap ] , mode='concat')
    

    x=  Dense(350, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model



def Dense_model_broad_reg(Inputs,nclasses,Inputshapes,dropoutRate=-1, npred = 1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """  
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform',input_shape=Inputshapes[0])(Inputs[0])
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Flatten()(vtx)
    
    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    x=  Dense(350, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = merge( [Inputs[4], x ] , mode='concat')
    predictions = Dense(npred, activation='linear',kernel_initializer='he_normal')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model





def Dense_model_broad_reg2(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    Flavour tagging and regression in one model. Fully working
    """
   
    
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Flatten()(vtx)
    
    
    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    

    x=  Dense(350, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    flav=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    
    
    
    
    ptcpf  = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    ptcpf = Dropout(dropoutRate)(ptcpf)
    ptcpf = Flatten()(ptcpf)
    ptnpf  = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(Inputs[2])
    ptnpf = Dropout(dropoutRate)(ptnpf)
    ptnpf = Flatten()(ptnpf)
   
    xx=merge( [Inputs[4],flav,ptcpf,ptnpf] , mode='concat')
    xx=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(xx)
    
    ptandsigma=Dense(2, activation='linear',kernel_initializer='lecun_uniform')(xx)
    
    predictions = [flav,ptandsigma]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def Dense_model_lessbroad(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """
   
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform',input_shape=Inputshapes[0])(Inputs[0])
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    
    
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Flatten()(vtx)
    
    
    x = merge( [Inputs[0],cpf,npf,vtx ] , mode='concat')
    x = Dropout(dropoutRate)(x)
    

    x=  Dense(600, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model






def Dense_model_microPF(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    from keras.layers.local import LocallyConnected1D
   
    #npf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    #npf = Dropout(dropoutRate)(npf)
    #npf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    #npf = Dropout(dropoutRate)(npf)
    #npf  = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    #npf = Dropout(dropoutRate)(npf)
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
    

def Dense_model_ConvCSV(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    Inputs similar to 2016 training, but with covolutional layers on each track and sv
    """
    
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    a = Dropout(dropoutRate)(a)
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(a)
    a = Dropout(dropoutRate)(a)
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(a)
    a = Dropout(dropoutRate)(a)
    a=Flatten()(a)
    
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    c = Dropout(dropoutRate)(c)
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(c)
    c = Dropout(dropoutRate)(c)
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(c)
    c = Dropout(dropoutRate)(c)
    c=Flatten()(c)
    
    x = merge( [Inputs[0],a,c] , mode='concat')
    
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
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

def binned3D_convolutional_classification_regression(inputs, output_shapes, dropout_rate=0.1, run_regression=True): 
    'Binned development output in [flavour, regression]'
    from keras.layers import Dense, Dropout, Flatten, Convolution2D, concatenate, \
       Convolution1D, Convolution3D, Reshape, LocallyConnected2D
    from keras.models import Model
    
    #unpack inputs
    glob, charged, neutral, svs, bin_global = inputs
    
    def make_layers(layer_type, args_list, dropout, 
                    layer_in, dropout_at_first=False, **kwargs):
        ret = layer_in
        first = not dropout_at_first
        for args in args_list:
            if first:
                first = False
            else:
                ret = Dropout(dropout)(ret)
            ret = layer_type(*args, **kwargs)(ret)
        return ret    
    
    #build single bin convolutions
    kwargs  = {'kernel_initializer' : 'lecun_uniform',  'activation' : 'relu'}
    k = (1,1,1) #kernel
    charged = make_layers(Convolution3D, [[64, k], [32, k], [32, k], [8, k]], dropout_rate, charged, **kwargs)
    svs     = make_layers(Convolution3D, [[64, k], [32, k], [32, k], [8, k]], dropout_rate, svs, **kwargs)
    neutral = make_layers(Convolution3D, [[32, k], [16, k], [4 , k]], dropout_rate, neutral, **kwargs)
    
    #flatten the single bins
    charged = Reshape((
          int(charged.shape[1]), # [0] is the number of batches, set to None, 1 is the # of x bins, returns Dimension(5), we need to cast to int
          int(charged.shape[2]), # 2 is the # of y bins
          int(charged.shape[3]*charged.shape[4]),
          ))(charged)
    neutral = Reshape((
          int(neutral.shape[1]), 
          int(neutral.shape[2]), 
          int(neutral.shape[3]*neutral.shape[4]),
          ))(neutral)
    svs = Reshape((
          int(svs.shape[1]), 
          int(svs.shape[2]), 
          int(svs.shape[3]*svs.shape[4]),
          ))(svs)
    
    for ib in [1,2]:
       if not (charged.shape[ib] == neutral.shape[ib] == svs.shape[ib]):
          raise ValueError('The number of bins along the axis %d should be consistent for charged, neutral and sv features' % ib)
    
    #merge the info from different sources, but from the same bin, into a single place
    binned_info = concatenate([charged,neutral,svs,bin_global]) #shape (?, 10, 10, 76)
    
    #shrink the info size
    nentries_per_bin = int(binned_info.shape[-1])
    k = (1,1)
    binned_info = make_layers(
        Convolution2D, [[nentries_per_bin//2, k], [10, k]], 
        dropout_rate, binned_info, 
        dropout_at_first=True, **kwargs
        )
    
    #learn a different representation for each bin
    #and reduce the dimensionality to 3x3
    xbins, ybins = int(binned_info.shape[1]), int(binned_info.shape[2])
    kernel = xbins//3
    if xbins != ybins:
        raise ValueError('The number of x and y bins should be the same!')
    if xbins % 3 != 0:
        raise ValueError('The number of x and y bins should be a multiplier of 3')
    
    binned_info = make_layers(
        LocallyConnected2D, [[40, kernel]], 
        dropout_rate, binned_info,
        dropout_at_first=True, strides=kernel,
        data_format='channels_last', kernel_initializer='lecun_uniform',
        activation='relu'
    )
    
    #
    # Dense part
    #
    
    binned_info = Reshape((
          int(binned_info.shape[1]) * 
          int(binned_info.shape[2]) * 
          int(binned_info.shape[3]), #coma to make it a tuple
          ))(binned_info) #Flatten()(binned_info) seems to fuck up the tensor output shape, so I reshape
    X = concatenate([glob, binned_info])
    
    X = make_layers(
        Dense, [[int(X.shape[1])]]+[[100]]*7, 
        dropout_rate, X,
        dropout_at_first=True, kernel_initializer='lecun_uniform',
        activation='relu'
    )
    X = Dropout(dropout_rate)(X)
    
    flavour = Dense(output_shapes[0], activation='softmax',kernel_initializer='lecun_uniform')(X)
    if run_regression:
        # regression
        pt_sigma = Dense(2, activation='linear', kernel_initializer='lecun_uniform')(X)
        # classification
        return Model(inputs=list(inputs), outputs=[flavour, pt_sigma])
    else:
        return Model(inputs=list(inputs), outputs=flavour)
    
