#!/usr/bin/env python3

print('importing tensorflow...')

import tensorflow
print(tensorflow.__file__)

print('importing DeepJetCore base')
import DeepJetCore
print(DeepJetCore.__file__)


print('importing DJC masked tf.keras as keras...')
import keras

from keras import Input

print('importing numpy...')

import numpy

print('running random training in keras...')

features = numpy.random.rand(1000, 10)
truth = features


a = Input(shape=(10,))
b = keras.layers.Dense(10)(a)
model = keras.models.Model(inputs=a, outputs=b)
model.compile(optimizer='adam',loss='mse')
model.fit(x=features, y=truth, batch_size=100, epochs=20)

print('loading DeepJetCore compiled library...')

from DeepJetCore.compiled import c_arrayReads

print('basic packages seem to be compiled... testing conversion')

import os
djc_base = os.environ.get('DEEPJETCORE')
script='''
#!/bin/bash
cd {djc_base}/testing
rm -rf batchDC 
export PYTHONPATH=`pwd`:$PYTHONPATH
echo $PYTHONPATH
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


print('tesing subpackage, training and prediction')

script='''
#!/bin/bash
cd {djc_base}/testing
rm -rf Subpackage
createSubpackage.py --data Subpackage
cd Subpackage

export SUBPACKAGE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$SUBPACKAGE

cd $SUBPACKAGE
export PYTHONPATH=$SUBPACKAGE/modules:$PYTHONPATH
export PYTHONPATH=$SUBPACKAGE/modules/datastructures:$PYTHONPATH
export PATH=$SUBPACKAGE/scripts:$PATH

export LD_LIBRARY_PATH=$SUBPACKAGE/modules/compiled:$LD_LIBRARY_PATH
export PYTHONPATH=$SUBPACKAGE/modules/compiled:$PYTHONPATH

cd example_data
convertFromSource.py -i train_files.txt -o conv -c TrainData_example
cd ../Train/
python3 training_example.py ../example_data/conv/dataCollection.djcdc  TEST
cd TEST
predict.py KERAS_model.h5 trainsamples.djcdc ../../example_data/test_files.txt PRED
'''.format(djc_base=djc_base)
os.system(script)






