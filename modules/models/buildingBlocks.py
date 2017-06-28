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
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        cpf = Dropout(dropoutRate)(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        cpf = Dropout(dropoutRate)(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        cpf = Dropout(dropoutRate)(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(8,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        npf = Dropout(dropoutRate)(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(4,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(8,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourDense(x,dropoutRate,active=True):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        x = Dropout(dropoutRate)(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x




def block_SchwartzImage(image,dropoutRate,active=True):
    '''
    returns flattened output
    '''
    
    if active:
        image =   Convolution2D(64, (8,8)  , border_mode='same', activation='relu',
                                kernel_initializer='lecun_uniform', name='swz_conv0')(image)
        image = MaxPooling2D(pool_size=(2, 2), name='swz_maxpool0')(image)
        image = Dropout(dropoutRate)(image)
        image =   Convolution2D(64, (4,4) , border_mode='same', activation='relu',
                                kernel_initializer='lecun_uniform', name='swz_conv1')(image)
        image = MaxPooling2D(pool_size=(2, 2), name='swz_maxpool1')(image)
        image = Dropout(dropoutRate)(image)
        image =   Convolution2D(64, (4,4)  , border_mode='same', activation='relu',
                                kernel_initializer='lecun_uniform', name='swz_conv2')(image)
        image = MaxPooling2D(pool_size=(2, 2), name='swz_maxpool2')(image)
        image = Dropout(dropoutRate)(image)
        image = Flatten()(image)

    else:
        #image=Cropping2D(crop)(image)#cut almost all of the 20x20 pixels
        image = Flatten()(image)
        image = Dense(1,kernel_initializer='zeros',trainable=False, name='swz_conv_off')(image)#effectively multipy by 0
        
    return image
