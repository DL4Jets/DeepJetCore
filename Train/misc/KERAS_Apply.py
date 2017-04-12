import keras
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import numpy.core.records
import sys

inputDataDir = sys.argv[1]
if inputDataDir[-1] != "/":
    inputDataDir+="/"
inputModelDir = sys.argv[2]
if inputModelDir[-1] != "/":
    inputModelDir+="/"

mymodel = load_model(inputModelDir+"KERAS_model.h5")
#x_local = np.load(inputDataDir+'local_X.npy')
x_global = np.load(inputDataDir+'global_X.npy')
predict_test_list = mymodel.predict( x_global)
#print 'shapes ' , predict_test_list[0].shape, ' ' , predict_test_list[1].shape
#predict_test = np.hstack((predict_test_list[0],predict_test_list[1])
#print x_global.shape

predict_write = np.core.records.fromarrays(  predict_test_list.transpose(), 
                                             names='probB, probC, probUDSG',
                                             formats = 'float32,float32,float32')


#reg_truth = np.load(inputDataDir+'regres_truth.npy')
#class_truth = np.load(inputDataDir+'class_truth.npy')
#class_truth = np.array(class_truth.tolist())
#print class_truth.type
from root_numpy import array2root
array2root(predict_write,inputModelDir+"KERAS_result_val.root",mode="recreate")
#array2root(reg_truth,inputModelDir+"re_truth.root",mode="recreate")
#array2root(class_truth,inputModelDir+"class_truth.root",mode="recreate")
