'''
Created on 20 Mar 2017

@author: jkiesele
'''


import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def predictAndMakeRoc(features_val, labels_val, nameprefix, names,formats, model):

    


    predict_test = model.predict(features_val)
    metric=model.evaluate(features_val, labels_val, batch_size=10000)
    
    print(metric)
    
    predict_write = np.core.records.fromarrays(  predict_test.transpose(), 
                                                 names=names,
                                                 formats = formats)
    
    # this makes you some ROC curves
    from sklearn.metrics import roc_curve
    
    # ROC one against all
    plt.figure(3)
    for i in range(labels_val.shape[1]):
    #    print (i , ' is', labels_val[i][:], ' ', predict_test[i][:])
        
        fpr , tpr, _ = roc_curve(labels_val[:,i], predict_test[:,i])
    #   print (fpr, ' ', tpr, ' ', _)
        plt.plot(tpr, fpr, label=predict_write.dtype.names[i])
    print (predict_write.dtype.names)
    plt.semilogy()
    plt.legend(predict_write.dtype.names, loc='upper left')
    plt.savefig(nameprefix+'ROCs.pdf')
    plt.close(3)
    
    # ROC one against som others
    plt.figure(4)
    # b vs light (assumes truth C is at index 1 and b truth at 0
    labels_val_noC = (labels_val[:,1] == 1)
    labels_val_killedC = labels_val[np.invert(labels_val_noC) ]
    predict_test_killedC = predict_test[np.invert(labels_val_noC)]
    fprC , tprC, _ = roc_curve(labels_val_killedC[:,0], predict_test_killedC[:,0])
    BvsL, = plt.plot(tprC, fprC, label='b vs. light')
    # b vs c (assumes truth light is at index 2
    labels_val_noL = (labels_val[:,2] ==1)
    
    labels_val_killedL = labels_val[np.invert(labels_val_noL)]
    predict_test_killedL = predict_test[np.invert(labels_val_noL)]
    fpr , tpr, _ = roc_curve(labels_val_killedL[:,0], predict_test_killedL[:,0])
    BvsC, = plt.plot(tpr, fpr, label='b vs. c')
    plt.semilogy()
    #plt.legend([BvsL,BvsC],loc='upper left')
    plt.ylabel('BKG efficiency')
    plt.xlabel('b efficiency')
    plt.ylim((0.001,1))
    plt.grid(True)
    plt.savefig(nameprefix+'ROCs_multi.pdf')
    plt.close(4)
    
    return metric
    