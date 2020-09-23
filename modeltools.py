from DeepJetCore.customObjects import *

custom_objs = get_custom_objects()
    
    
def getLayer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
        


def printLayerInfosAndWeights(model, noweights=False):
    for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print (g)
        if noweights: continue
        print (h)


def fixLayersContaining(m, fixOnlyContaining, invert=False):
    import collections.abc
    if isinstance(fixOnlyContaining, collections.abc.Sequence) and not isinstance(fixOnlyContaining, str):
        isseq=True
    else:
        isseq=False
    if not isseq:
        fixOnlyContaining=[fixOnlyContaining]
    if invert:
        for layidx in range(len(m.layers)):
            m.get_layer(index=layidx).trainable=False
        for layidx in range(len(m.layers)):
            for ident in fixOnlyContaining:
                if len(ident) and ident in m.get_layer(index=layidx).name:
                    m.get_layer(index=layidx).trainable=True
    else:
        for layidx in range(len(m.layers)):
            for ident in fixOnlyContaining:    
                if len(ident) and ident in m.get_layer(index=layidx).name:
                    m.get_layer(index=layidx).trainable=False
    return m

def set_trainable(m, patterns, value):
    if isinstance(patterns, basestring):
        patterns = [patterns]
    for layidx in range(len(m.layers)):
        name = m.get_layer(index=layidx).name
        if any(i in name for i in patterns):
            m.get_layer(index=layidx).trainable = value
    return m

def setAllTrainable(m, val=True):
    for layidx in range(len(m.layers)):
        m.get_layer(index=layidx).trainable = val
    return m

def loadModelAndFixLayers(filename,fixOnlyContaining):
    #import keras
    from keras.models import load_model
    
    m=load_model(filename, custom_objects=custom_objs)
    
    fixLayersContaining(m, fixOnlyContaining)
                
    return m

def load_model(filename):
    from keras.models import load_model
    
    model=load_model(filename, custom_objects=custom_objs)
    
    return model

def apply_weights_where_possible(target_model, weight_model):
    
    for layer_a in target_model.layers:
        for layer_b in weight_model.layers:
            if layer_a.name == layer_b.name:
                try:
                    layer_a.set_weights(layer_b.get_weights()) 
                    print('using weights from ',  layer_a.name)
                except:  
                    print('unable to copy weights for layer ',  layer_a.name)
                    #print(layer_a.weights,'\n',layer_b.weights)
    
    
    return target_model





################# wrappers for keras models in DJC

import tensorflow as tf

class DJCKerasModel(tf.keras.models.Model):
    '''
    Base class to implement automatic shape feeding as in DJC
    Interfaces smoothly with training_base
    '''
    def __init__(self,*args,**kwargs):
        
        super(DJCKerasModel, self).__init__(*args,dynamic=False,**kwargs)
        self.keras_input_shapes=None
        self._is_djc_keras_model = True
    
    def setInputShape(self,keras_inputs):
        self.keras_input_shapes=[i.shape for i in keras_inputs]
        
    def build(self,input_shapes):
        super(DJCKerasModel,self).build(self.keras_input_shapes)
    












