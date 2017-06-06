#!/usr/bin/env python

from keras.models import load_model
from testing import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
from Losses import loss_NLL
import os


parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
parser.add_argument('loss', nargs='?', default='keras_loss')

args = parser.parse_args()

 
if os.path.isdir(args.outputDir):
    print('output directory must not exists yet')
    raise Exception('output directory must not exists yet')

if(args.loss=='loss_NLL'):
    print ('using custom loss loss_NLL')
    model=load_model(args.inputModel, custom_objects={'loss_NLL':loss_NLL})
else:
    model=load_model(args.inputModel)

td=testDescriptor()

from DataCollection import DataCollection

testd=DataCollection()
testd.readFromFile(args.inputDataCollection)


os.mkdir(args.outputDir)

td.makePrediction(model, testd, args.outputDir)

td.writeToTextFile(args.outputDir+'/tree_association.txt')

#    make the file reading entirely C++
#    then it can be used for other studies
