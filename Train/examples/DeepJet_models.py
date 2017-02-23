from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, merge, Input
from keras.models import Model

def Incept_model(inputs,dropoutRate=0.25):
    """
        This NN adds two inputs, one for a conv net and a seceond for a dense net, both nets get combined. The last layer is split into regression and classification activations (softmax, linear)
    """
    
    x =   Convolution2D(20, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(inputs[0])
    # add more layers to get deeper
    x =   Convolution2D(10, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(x)
    x =   Convolution2D(5, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(x)
    x = Flatten()(x)
    #  Here add e.g. the normal dense stuff from DeepCSV
    y = Dense(1, activation='relu',init='lecun_uniform',input_shape=(1,))(inputs[1])
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
    model = Model(input=inputs, output=predictions)
    return model

def Dense_model(inputs,nclasses,dropoutRate=0.1):
    """
    Dense matrix, defaults similat to 2016 training
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(100, activation='relu',init='lecun_uniform',input_shape=(66,))(inputs)
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    # x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    # x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',init='lecun_uniform')(x)
    # x = Dropout(dropoutRate)(x)
    # x=  Dense(100, activation='relu',init='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',init='lecun_uniform')(x)
    model = Model(input=inputs, output=predictions)
    return model
