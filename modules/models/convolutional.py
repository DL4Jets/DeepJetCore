from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM,merge, Convolution1D, Conv2D
from keras.models import Model
 
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourDense, block_SchwartzImage

def convolutional_model_broad(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour', as for DPS note
    """  
   
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    cpf = Flatten()(cpf)
    npf = Flatten()(npf)
    vtx = Flatten()(vtx)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ])
    
    x  = block_deepFlavourDense(x,dropoutRate)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad_map(Inputs,nclasses,Inputshapes,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2)(cpf)
    npf = LSTM(50,go_backwards=True,implementation=2)(npf)
    vtx = LSTM(50,go_backwards=True,implementation=2)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def convolutional_model_broad_map_reg(Inputs,nclasses,Inputshapes,dropoutRate,npred):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2)(cpf)
    npf = LSTM(50,go_backwards=True,implementation=2)(npf)
    vtx = LSTM(50,go_backwards=True,implementation=2)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    predictions = [Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x),
                   Dense(npred, activation='linear',kernel_initializer='ones')(x)]
    model = Model(inputs=Inputs, outputs=predictions)
    return model




def convolutional_model_broad_reg(Inputs,nclasses,Inputshapes,dropoutRate=-1, npred = 1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """  
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    cpf = Flatten()(cpf)
    npf = Flatten()(npf)
    vtx = Flatten()(vtx)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    x = Concatenate()( [Inputs[4], x ] )
    predictions = Dense(npred, activation='linear',kernel_initializer='he_normal')(x),
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad_reg2(Inputs,nclasses,Inputshapes,dropoutRate=-1):
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
    
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ] )
    

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
   
    xx=Concatenate()( [Inputs[4],flav,ptcpf,ptnpf] )
    xx=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(xx)
    
    ptandsigma=Dense(2, activation='linear',kernel_initializer='lecun_uniform')(xx)
    
    predictions = [flav,ptandsigma]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_lessbroad(Inputs,nclasses,Inputshapes,dropoutRate=-1):
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
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ] )
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

def convolutional_model_ConvCSV(Inputs,nclasses,Inputshape,dropoutRate=0.25):
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
    
    x = Concatenate()( [Inputs[0],a,c] )
    
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
