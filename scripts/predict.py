#!/usr/bin/env python

from keras.models import load_model
from testing import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
import os


parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
parser.add_argument('--labels', action='store_true', help='store true labels in the trees')
parser.add_argument('--monkey_class', default='', help='allows to read the data with a different TrainData, it is actually quite dangerous if you do not know what you are doing')

args = parser.parse_args()

 
if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')


model=load_model(args.inputModel, custom_objects=global_loss_list)


td=testDescriptor()

from DataCollection import DataCollection

testd=DataCollection()
testd.readFromFile(args.inputDataCollection)


os.mkdir(args.outputDir)

td.makePrediction(
    model, testd, args.outputDir,
    store_labels = args.labels,
    monkey_class = args.monkey_class
)

td.writeToTextFile(args.outputDir+'/tree_association.txt')

#    make the file reading entirely C++
#    then it can be used for other studies
