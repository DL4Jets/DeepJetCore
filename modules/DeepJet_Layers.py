


from keras.layers.normalization import BatchNormalization


class FixableBatchNormalization(BatchNormalization):
    '''
    Minimal extension of batch norm layer, such that trainable=False prevents all updates of batch norm, even in training.
    '''
    
    def __init__(self,**kwargs):
        BatchNormalization.__init__(self,**kwargs)
        
        
    def call(self, inputs, training=None):
        
        if self.trainable:
            return BatchNormalization.call(self,inputs=inputs, training=training)
        else:
            return BatchNormalization.call(self,inputs=inputs, training=False)