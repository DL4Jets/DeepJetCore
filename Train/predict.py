

from keras.models import load_model
from testing import testDescriptor
from argparse import ArgumentParser
import os

parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
args = parser.parse_args()


if os.path.isdir(args.outputDir):
    print('output directory must not exists yet')
    raise Exception('output directory must not exists yet')


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