
from DeepJetCore.compiled.c_simpleArray import simpleArray
from DeepJetCore.TrainData import TrainData
import numpy as np
import time
import copy
from DeepJetCore.TrainData import TrainData

data = np.arange(0, 64 , dtype='float32')
data = np.reshape(data, [-1,2])

rowsplits   = np.array([0, 2, 3, 7, 8, 11, 18, 19, 23, 25, 27, len(data)], dtype='int64')

truth = np.arange(0, len(rowsplits)-1 , dtype='float32')

batchsizes  = [3,
               5,
               10]

expected_elmts = [[2,None,1,1,None,1,None,1,1,None],
                  [2,2,1,None,2,2,1],
                  [4,2,4,1]
            ]

#print(data)


#arr.readFromFile("testfile.djcd")


#id, rs = arr.transferToNumpy()

print(data)
print(rowsplits)


#arr.readFromFile("testfile_plain.djcd")
#
#id, rs = arr.transferToNumpy()
#
#print(id)
#print(rs)
#
print("now for the creation, see if it checks out")
arr = simpleArray()

#check for mem leaks
arr.createFromNumpy(data, rowsplits)#), rowsplits)

trutharr = simpleArray()
trutharr.createFromNumpy(truth, np.array([]))

truth_array = simpleArray()
truth_array.createFromNumpy(truth, np.array([]))


arr.cout()

td = TrainData()
td.storeFeatureArray(arr)
td.storeTruthArray(truth_array)


print('td.nElements()',td.nElements())


filenames = ["file1.djctd"]
for f in filenames:
    td.writeToFile(f)
  
print("check generator")
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator




for b in range(len(batchsizes)):
    batchsize=batchsizes[b]
    expected_here = expected_elmts[b]
    
    gen = trainDataGenerator()
    gen.debug=True
    #gen.setSquaredElementsLimit(True)
    gen.setBatchSize(batchsize)
    print('batchsize',batchsize)
    print('reading in info')
    gen.setFileList(filenames)
    print('done reading in')
    

    #print("expect: 5, 4, 4, 5, 4, 4 = 6 ")
    nbatches = gen.getNBatches()
    #print(nbatches)
    print("gen.getNTotal()", gen.getNTotal()) # <- bullshit
    print("gen.getNBatches()", gen.getNBatches())
    
    
    #gen.debug=True
    i_e = 0
    for i in range(nbatches):
        print("batch", i, "is last ", gen.lastBatch())
        d = gen.getBatch()
        nelems=d.nElements()
        if expected_here[i_e] is None:
            i_e += 1
        assert expected_here[i_e] == nelems
        i_e += 1


print('testing direct buffering')


        
for b in range(len(batchsizes)):
    batchsize=batchsizes[b]
    expected_here = expected_elmts[b]
    
    gen = trainDataGenerator()
    gen.debug=True
    #gen.setSquaredElementsLimit(True)
    gen.setBatchSize(batchsize)
    print('batchsize',batchsize)
    print('setting buffer')
    gen.setBuffer(td)
    print('done setting buffer')
    

    #print("expect: 5, 4, 4, 5, 4, 4 = 6 ")
    nbatches = gen.getNBatches()
    #print(nbatches)
    print("gen.getNTotal()", gen.getNTotal()) # <- bullshit
    print("gen.getNBatches()", gen.getNBatches())
    
    
    #gen.debug=True
    i_e = 0
    for i in range(nbatches):
        print("batch", i, "is last ", gen.lastBatch())
        d = gen.getBatch()
        nelems=d.nElements()
        if expected_here[i_e] is None:
            i_e += 1
        assert expected_here[i_e] == nelems
        i_e += 1




print('checking 1 example generator')

td.skim(0)
gen = trainDataGenerator()
gen.setBuffer(td)
d = gen.getBatch()
nelems=d.nElements()

assert nelems == 1















