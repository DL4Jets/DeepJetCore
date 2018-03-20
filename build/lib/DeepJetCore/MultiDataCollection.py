from DataCollection import DataCollection
from multiprocessing import cpu_count
from itertools import izip
from pdb import set_trace
import numpy as np
from copy import deepcopy

class MultiDataCollection(object):
  '''This class allows the simultaneous use of multiple
  DataCollections for a training, it provides the same interface
  and adds the functionality of adding Y targets as well as flags
  returned instead of the weights.
  In case of weights the flag is multiplied by the weight value
  Constructor ([infiles = None[, nprocs = -1[, add_ys = [][, flags=[]]]]])
  optional parameters:
  infiles: list of input dataCollection files to be opened
  nprocs: number of processors to use
  add_ys: list of additional Y targets to be added at generator time, 
  must be the same length of the input collections. The list content 
  must be iterable, each iteration produces a new target, 
  only scalar type supprted for now
  flags: like add_ys, same rules apply. The flags gets multiplied 
  by the event weight is case weights are used. The lenght of the 
  flags must be the same as the TOTAL number of Y targets. 
  Flags are returned instead of the event weights
  '''
  def __init__(self, infiles = None, nprocs = -1, add_ys = [] ,flags=[]):
    '''Constructor'''
    self.collections = []
    self.nprocs = nprocs       
    self.meansnormslimit=500000 
    self.flags = []
    self.generator_modifier = lambda x: x
    self.additional_ys = []
    if infiles:
      self.collections = [
        DataCollection(
          i, 
          cpu_count()/len(infiles) if nprocs == -1 else nprocs/len(infiles)
        ) for i in infiles]
    if flags:
      self.setFlags(flags)
    if add_ys:
      self.addYs(add_ys)

  @property
  def useweights(self):
    return all(i.useweights for i in self.collections)

  @useweights.setter
  def useweights(self, val):
    for i in self.collections:
      i.useweights = val

  def addYs(self, add_ys):
    'adds Ys that will be appended on the fly to the generator, Ys are a list of iterables'
    if len(add_ys) != len(self.collections):
      raise ValueError('The Ys must be the same lenght of the input collections')
    self.additional_ys = add_ys

  def readFromFile(self, infiles):
    self.collections = [
      DataCollection(
        i, cpu_count()/len(infiles) if self.nprocs == -1 else self.nprocs/len(infiles)
      ) for i in infiles]

  def setFlags(self, flags):
    'adds flags that will be added on the fly to the generator, flags are a list of iterables'
    if len(flags) != len(self.collections):
      raise ValueError('The flags must be the same lenght of the input collections')
    self.flags = flags

  def getInputShapes(self):
    'Gets the input shapes from the data class description'
    shapes = [i.getInputShapes() for i in self.collections]
    if not all(i == shapes[0] for i in shapes):
      raise ValueError('Input collections have different input shapes!')
    return shapes[0]
    
  def getTruthShape(self):
    shapes = [i.getTruthShape() for i in self.collections]
    if not all(i == shapes[0] for i in shapes):
      raise ValueError('Input collections have different input shapes!')
    return shapes[0]
        
  def getNRegressionTargets(self):
    shapes = [i.getNRegressionTargets() for i in self.collections]
    if not all(i == shapes[0] for i in shapes):
      raise ValueError('Input collections have different input shapes!')
    return shapes[0]
    
  def getNClassificationTargets(self):
    shapes = [i.getNClassificationTargets() for i in self.collections]
    if not all(i == shapes[0] for i in shapes):
      raise ValueError('Input collections have different input shapes!')
    return shapes[0]
        
  def getUsedTruth(self):
    shapes = [i.getUsedTruth() for i in self.collections]
    if not all(i == shapes[0] for i in shapes):
      raise ValueError('Input collections have different input shapes!')
    return shapes[0]
  
  def split(self,ratio):
    'splits the sample into two parts, one is kept as the new collection, the other is returned'
    out = [i.split(ratio) for i in self.collections]
    retval = deepcopy(self)
    retval.collections = out
    return retval

  def writeToFile(self, fname):
    for idx, i in enumerate(self.collections):
      i.writeToFile(fname.replace('.dc', '%d.dc' % idx))    

  def generator(self):
    '''Batch generator. Heavily based on the DataCollection one. 
Adds flags on the fly at the end of each Y'''
    generators = [i.generator() for i in self.collections]
    flags = self.flags if self.flags else [None for i in self.collections]
    add_ys = self.additional_ys if self.additional_ys else [[] for i in self.collections]
    for zipped in izip(*generators):
      xtot, wtot, ytot = None, None, None
      for xyw, flag, add_y in zip(zipped, flags, add_ys):
        if len(xyw) == 3:
          x, y, w = deepcopy(xyw)
        else: #len(xyw) == 3:
          x, y = deepcopy(xyw)
          w = [np.ones((x[0].shape[0]))] if self.flags else None

        batch_size = x[0].shape[0]
        ones = np.ones((batch_size, 1))
        for template in add_y:
          y_to_add = np.hstack([ones*i for i in template]) \
             if hasattr(template, '__iter__') else \
             ones*template
          y.append(y_to_add)          

        #create the flags
        if self.flags:
          if len(flag) != len(y):
            raise ValueError(
              'Flags (if any) and total Y number MUST'
              ' be the same! Got: %d and %d' % (len(flag), len(y)))       
          w = [w[0]*i for i in flag]
        
        if xtot is None:
          xtot = x
          ytot = y
          wtot = w
        else:
          xtot = [np.vstack([itot, ix]) for itot, ix in zip(xtot, x)]
          ytot = [np.vstack([itot, iy]) for itot, iy in zip(ytot, y)]
          if w is not None:
            wtot = [np.concatenate([itot, iw]) for itot, iw in zip(wtot, w)]

      if wtot is None:
        yield self.generator_modifier((xtot, ytot))
      else:
        yield self.generator_modifier((xtot, ytot, wtot))
    
  def __len__(self):
    return sum(len(i) for i in self.collections)

  @property
  def sizes(self):
    return [len(i) for i in self.collections]

  @property
  def nsamples(self):
    return len(self)

  def setBatchSize(self,bsize):
    if bsize > len(self):
      raise Exception('Batch size must not be bigger than total sample size')
    for i in self.collections:
      batch = bsize*len(i)/len(self)
      i.setBatchSize(batch)

  @property
  def batches(self):
    return [i.batch_size for i in self.collections]

  def getAvEntriesPerFile(self):
    return min(i.getAvEntriesPerFile() for i in self.collections)

  @property
  def maxFilesOpen(self):
    return max(i.maxFilesOpen for i in self.collections)

  @maxFilesOpen.setter
  def maxFilesOpen(self, val):    
    for i in self.collections:
      i.maxFilesOpen = val

  def getNBatchesPerEpoch(self):
    return sum(i.getNBatchesPerEpoch() for i in self.collections)/len(self.collections)
  


