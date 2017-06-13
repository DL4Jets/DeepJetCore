from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D
from keras.models import Model    

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
