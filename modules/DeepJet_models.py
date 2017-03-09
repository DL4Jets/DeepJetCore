#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D
from keras.models import Model

#fix for dropout on gpus
import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops


def Incept_model(Inputs,dropoutRate=0.25):
    """
        This NN adds two inputs, one for a conv net and a seceond for a dense net, both nets get combined. The last layer is split into regression and classification activations (softmax, linear)
    """
    
    x =   Convolution2D(20, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(Inputs[0])
    # add more layers to get deeper
    x =   Convolution2D(10, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(x)
    x =   Convolution2D(5, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(x)
    x = Flatten()(x)
    #  Here add e.g. the normal dense stuff from DeepCSV
    y = Dense(1, activation='relu',init='lecun_uniform',input_shape=(1,))(Inputs[1])
    y = Dense(1, activation='relu',init='lecun_uniform')(y)
    y = Dense(1, activation='relu',init='lecun_uniform')(y)
    # add more layers to get deeper
    
    # combine convolutional and dense (global) layers
    x = merge( [x,y ] , mode='concat')
    
    # linear activation for regression and softmax for classification
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    # add more layers to get deeper
    
    predictions = [Dense(1, activation='linear',init='normal')(x),Dense(5, activation='softmax',init='lecun_uniform')(x)]
    model = Model(input=Inputs, output=predictions)
    return model

def Dense_model(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    Dense matrix, defaults similat to 2016 training
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',init='lecun_uniform',input_shape=Inputshape)(Inputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',init='lecun_uniform')(x)
    model = Model(input=Inputs, output=predictions)
    return model


def Dense_model2(Inputs,nclasses,Inputshape,dropoutRate=0.25):
    """
    Dense matrix, defaults similat to 2016 training
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(200, activation='relu',init='lecun_uniform',input_shape=Inputshape)(Inputs)
    x = Dense(200, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(200, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(200, activation='relu',init='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',init='lecun_uniform')(x)
    model = Model(input=Inputs, output=predictions)
    return model


def Dense_model_broad(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """
   
    gl = Dense(20, activation='relu',init='lecun_uniform',input_shape=Inputshapes[0])(Inputs[0])
    gl = Dense(20, activation='relu',init='lecun_uniform')(gl)
    gl = Dense(20, activation='relu',init='lecun_uniform')(gl)
    
    
    cpf  = Convolution1D(64, 1, init='lecun_uniform',  activation='relu')(Inputs[1])
    cpf  = Convolution1D(32, 1, init='lecun_uniform',  activation='relu')(cpf)
    cpf  = Convolution1D(24, 1, init='lecun_uniform',  activation='relu')(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(64, 1, init='lecun_uniform',  activation='relu',input_shape=Inputshapes[2])(Inputs[2])
    npf = Convolution1D(32, 1, init='lecun_uniform',  activation='relu')(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(64, 1, init='lecun_uniform',  activation='relu',input_shape=Inputshapes[3])(Inputs[3])
    vtx = Convolution1D(32, 1, init='lecun_uniform',  activation='relu')(vtx)
    vtx = Flatten()(vtx)
    
    
    
    x = merge( [cpf,npf ] , mode='concat')
    x = merge( [x,gl ] , mode='concat')
    x = merge( [x,vtx ] , mode='concat')
    
    
    
    x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(50, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(50, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(50, activation='relu',init='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(30, activation='relu',init='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',init='lecun_uniform')(x)
    model = Model(input=Inputs, output=predictions)
    return model

