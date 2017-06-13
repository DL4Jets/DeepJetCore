'''
standardised building blocks for the models
'''

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D
from keras.layers.pooling import MaxPooling2D


def block_deepFlavourConvolutions(charged,neutrals,vertices,dropoutRate,active=True):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
        cpf = Dropout(dropoutRate)(cpf)
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    else:
        cpf = Convolution1D(8,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    else:
        npf = Convolution1D(4,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    else:
        vtx = Convolution1D(8,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourDense(x,dropoutRate,active=True):
    if active:
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
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False)(x)
    
    return x




def block_SchwartzImage(image,dropoutRate,active=True):
    '''
    returns flattened output
    '''
    
    if active:
        image =   Convolution2D(64, (8,8)  , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(image)
        image = MaxPooling2D(pool_size=(2, 2))(image)
        image = Dropout(dropoutRate)(image)
        image =   Convolution2D(64, (4,4) , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(image)
        image = MaxPooling2D(pool_size=(2, 2))(image)
        image = Dropout(dropoutRate)(image)
        image =   Convolution2D(64, (4,4)  , border_mode='same', activation='relu',kernel_initializer='lecun_uniform')(image)
        image = MaxPooling2D(pool_size=(2, 2))(image)
        image = Dropout(dropoutRate)(image)
        image = Flatten()(image)

    else:
        #image=Cropping2D(crop)(image)#cut almost all of the 20x20 pixels
        image = Flatten()(image)
        image = Dense(1,kernel_initializer='zeros',trainable=False)(image)#effectively multipy by 0
        
    return image
