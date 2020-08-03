
djc_global_layers_list={}

from keras.layers import Layer
import tensorflow as tf



class StopGradient(Layer):
    def __init__(self, **kwargs):
        super(StopGradient, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.stop_gradient(inputs)

djc_global_layers_list['StopGradient']=StopGradient   

class SelectFeatures(Layer):
    def __init__(self, index_left, index_right, **kwargs):
        super(SelectFeatures, self).__init__(**kwargs)
        self.index_left=index_left
        self.index_right=index_right
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.index_right-self.index_left,)
    
    def call(self, inputs):
        
        return inputs[...,self.index_left:self.index_right]
    
    def get_config(self):
        config = {'index_left': self.index_left,'index_right': self.index_right}
        base_config = super(SelectFeatures, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    
    
djc_global_layers_list['SelectFeatures']=SelectFeatures


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
    

class Clip(Layer):
    def __init__(self, min, max , **kwargs):
        super(Clip, self).__init__(**kwargs)
        self.min=min
        self.max=max
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)
    
    def get_config(self):
        config = {'min': self.min, 'max': self.max}
        base_config = super(Clip, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    
djc_global_layers_list['Clip']=Clip


class ReduceSumEntirely(Layer):
    def __init__(self,  **kwargs):
        super(ReduceSumEntirely, self).__init__(**kwargs)
        
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)
    
    def call(self, inputs):
        red_axes=(inputs.shape[1:]).as_list()
        red_axes = [i+1 for i in range(len(red_axes))]
        return tf.expand_dims(tf.reduce_sum(inputs,axis=red_axes),axis=1)
    
    def get_config(self):
        base_config = super(ReduceSumEntirely, self).get_config()
        return dict(list(base_config.items()))
    

djc_global_layers_list['ReduceSumEntirely']=ReduceSumEntirely