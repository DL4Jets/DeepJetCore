import numpy
import sys
import os

"""
Some math on the truth
"""

inputDataName = sys.argv[1]

truth = numpy.load(inputDataName)
print (type(truth),truth.shape)
b = truth['isB'].view(numpy.ndarray)
c = truth['isC'].view(numpy.ndarray)
uds = truth['isUDS'].view(numpy.ndarray)
g = truth['isG'].view(numpy.ndarray)
l = g + uds
all = numpy.vstack((b,c,l)).transpose()
newfile = inputDataName[0:-4]
print (all.shape)
newfile += '3.npy'
print (newfile)
numpy.save(newfile ,all)
# print (all.shape)
