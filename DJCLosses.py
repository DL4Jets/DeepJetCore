
djc_global_loss_list = {}

import tensorflow as tf

def dummy_loss(truth, pred):
    t = tf.Print(truth,[truth],'truth ')
    p = tf.Print(pred,[pred],'pred ')
    return tf.reduce_mean(t)+tf.reduce_mean(p)

djc_global_loss_list['dummy_loss']=dummy_loss