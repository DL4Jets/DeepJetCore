
djc_global_layers_list={}

from keras.layers import Layer
import tensorflow as tf

class ScalarMultiply(Layer):
    def __init__(self, factor, **kwargs):
        super(ScalarMultiply, self).__init__(**kwargs)
        self.factor=factor
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return inputs*self.factor
    
    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(ScalarMultiply, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    
    
djc_global_layers_list['ScalarMultiply']=ScalarMultiply

class Print(Layer):
    def __init__(self, message, **kwargs):
        super(Print, self).__init__(**kwargs)
        self.message=message
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.Print(inputs,[inputs],self.message,summarize=300)
    
    def get_config(self):
        config = {'message': self.message}
        base_config = super(Print, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    

djc_global_layers_list['Print']=Print

### the following ones should go to DeepJetCore

class ReplaceByNoise(Layer):
    def __init__(self, **kwargs):
        super(ReplaceByNoise, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.random_normal(shape=tf.shape(inputs),
                                mean=0.0,
                                stddev=1.0,
                                dtype='float32')
        
    
    def get_config(self):
        base_config = super(ReplaceByNoise, self).get_config()
        return dict(list(base_config.items()))
    

djc_global_layers_list['ReplaceByNoise']=ReplaceByNoise
    
    

class FeedForward(Layer):
    def __init__(self, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return 1.*inputs
        
    
    def get_config(self):
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()))
    

djc_global_layers_list['FeedForward']=FeedForward
    
    
