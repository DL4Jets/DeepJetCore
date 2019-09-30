#!/usr/bin/env python

print('importing tensorflow...')

import tensorflow

print('importing keras...')

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

print('loading DeepJetCore library...')

from DeepJetCore.compiled import c_arrayReads

print('basic packages seem to work')
