
djc_global_loss_list = {}

import tensorflow as tf

def dummy_loss(truth, pred):
    #t = tf.Print(truth,[truth],'truth ')
    #p = tf.Print(pred,[pred],'pred ')
    return (tf.reduce_mean(truth)-tf.reduce_mean(pred))**2

djc_global_loss_list['dummy_loss']=dummy_loss