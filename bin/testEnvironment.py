#!/usr/bin/env python3

print('importing tensorflow...')

import tensorflow
print(tensorflow.__file__)

print('importing DeepJetCore base')
import DeepJetCore
print(DeepJetCore.__file__)
import os
djc_base = os.environ.get('DEEPJETCORE')


print('importing DJC masked tf.keras as keras...')
import keras

from keras import Input

print('importing numpy...')

import numpy

print('running random training in keras...')

def gen():
    while(True):
        features = numpy.random.rand(100, 10)
        truth = features
        yield (features,truth)


a = [Input(shape=(10,))]
b = [keras.layers.Dense(10)(a[0])]
model = keras.models.Model(inputs=a, outputs=b)
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(x=gen(), steps_per_epoch=100, epochs=3)


print('loading DeepJetCore compiled library...')

from DeepJetCore.compiled import c_arrayReads

print('basic packages seem to be compiled... testing conversion')


if True:
    script='''
    #!/bin/bash
    cd {djc_base}/testing
    rm -rf batchDC 
    export PYTHONPATH=`pwd`:$PYTHONPATH
    convertFromSource.py -i files/filelist.txt -o batchDC -c TrainData_testBatch -n 1
    '''.format(djc_base=djc_base)
    os.system(script)
    
    print('testing batch explosion. Please check batch loss plot afterwards for smoothness. Warnings about the callback time can be ignored.')

script='''
#!/bin/bash
cd {djc_base}/testing
rm -rf batchExplode
export PYTHONPATH=`pwd`:$PYTHONPATH
python3 batch_explosion.py batchDC/dataCollection.djcdc batchExplode
'''.format(djc_base=djc_base)
os.system(script)






