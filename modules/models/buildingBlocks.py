'''
standardised building blocks for the models
'''

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

def block_deepFlavourBTVConvolutions(charged,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,vtx


def block_deepFlavourConvolutions(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
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
