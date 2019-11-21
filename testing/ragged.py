
import tensorflow as tf
#tf.enable_eager_execution()


#B x V x F
ragged_val_list = [ [[[1., 3.], [2, 4.], [3, 4.]], 
                                 [[1, 2.], [2, 3.]], 
                                 [[1, 2.], [2, 3.], [3, 4.], [4, 5.]]] ]



flat_list = tf.ragged.constant( 2*[[[[1.]]]] ).to_tensor()

#ragged_list = tf.ragged.constant(2*ragged_val_list).to_tensor()
truth_ragged_tensor = tf.ragged.constant(2*ragged_val_list).to_tensor()



def generator():
    while(1):
        yield (flat_list, [ flat_list,flat_list ]) # just returns the same every time
        #yield (ragged_list, truth_ragged_tensor) 


inputs_ragged = tf.keras.layers.Input(shape=(None, None,1), ragged=False)
outputs_ragged = tf.keras.layers.Dense(1)(inputs_ragged)
outputs_ragged2 = tf.keras.layers.Dense(1)(inputs_ragged)
model_ragged = tf.keras.Model(inputs=inputs_ragged, outputs=[outputs_ragged,outputs_ragged2])


globaltensor=None

def lossa(truth, pred):
    global globaltensor
    if globaltensor is not None:
        return ( tf.reduce_mean(truth) - tf.reduce_mean(globaltensor) + tf.reduce_mean(pred) )**2
    globaltensor= truth
    return 0.*tf.reduce_mean(pred)
    

def lossb(truth, pred):
    global globaltensor
    if globaltensor is not None:
        return ( tf.reduce_mean(truth) - tf.reduce_mean(globaltensor) + tf.reduce_mean(pred) )**2
    globaltensor= truth
    return 0.*tf.reduce_mean(pred)


model_ragged.compile(optimizer='Adam', loss = [lossa,lossb])#this could be a ragged one, from somewhere..


model_ragged.fit_generator(generator = generator(), steps_per_epoch=50, epochs=2, 
                           validation_data=generator(), validation_steps=2,
                           max_queue_size=1,
                           workers=0)






exit()
flatish = tf.constant([
    [1,0.1,0.01],
    [2,0.2,0.02],
    [3,0.3,0.03],
    [4,0.4,0.04],
    [5,0.5,0.05]
    ])

splits = tf.constant([
    0, 2, 3, 5
    ])

rt = tf.RaggedTensor.from_row_splits(
    values=flatish,
    row_splits=splits)
print(rt)
lrt = rt.to_list()
print(lrt)

for i in lrt:
    print(i)
    
## probably will be supported in one of the next TF releases
# https://github.com/tensorflow/tensorflow/issues/27170