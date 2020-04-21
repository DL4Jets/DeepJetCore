
from DeepJetCore.compiled.c_simpleArray import simpleArray
import numpy as np

data = np.arange(0, 64 , dtype='float32')
data = np.reshape(data, [-1,2])
rowsplits   = np.array([0, 2, 3, 7, 8, 11, 18, 19, 23, 25, 27, 32], dtype='int64')

arr = simpleArray()
arr.createFromNumpy(data, rowsplits)

slice = arr.getSlice(2,6)
nps,rss = slice.copyToNumpy(False)
nps_exp = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],dtype='float32')
nps_exp = np.reshape(nps_exp, [-1,2])
rss_exp = np.array([0,4,5,8,15],dtype='int64')

assert np.all(nps_exp == nps) and np.all(rss_exp == rss)

arr2 = simpleArray()
arr2.createFromNumpy(data, rowsplits)

arr.append(arr2)


arr3 = arr.split(11)

np1,rs1 = arr.copyToNumpy(False)
np3,rs3 = arr3.copyToNumpy(False)

assert np.all(np1 == np3) and np.all(np1 == data)
assert np.all(rs1 == rs3) and np.all(rs1 == rowsplits)

arr4 = arr.split(1)
np4,rs4 = arr.copyToNumpy(False)
arrexp_np4 = np.arange(4, 64 , dtype='float32')
arrexp_np4 = np.reshape(arrexp_np4, [-1,2])
rsexp_rs4  = np.array([2, 3, 7, 8, 11, 18, 19, 23, 25, 27, 32], dtype='int64')-2


np5,rs5 = arr4.copyToNumpy(False)
exp_np5 = np.arange(0, 4 , dtype='float32')
exp_np5 = np.reshape(exp_np5, [-1,2])
exp_rs5 = np.array([0, 2], dtype='int64')


assert np.all(np5 == exp_np5) and np.all(rs5 == exp_rs5)

arr.append(arr4)
np6,rs6 = arr.copyToNumpy(False)
exp_np6 = np.concatenate( [np.arange(4, 64 , dtype='float32'), np.arange(0, 4 , dtype='float32')], axis=-1)
exp_np6 = np.reshape(exp_np6, [-1,2])
exp_rs6 = np.concatenate([rsexp_rs4, np.array([32], dtype='int64')], axis=-1)

assert np.all(np6 == exp_np6) and np.all(rs6 == exp_rs6)

