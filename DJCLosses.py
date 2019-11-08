
djc_global_loss_list = {}

import tensorflow as tf

def dummy_loss(truth, pred):
    return tf.reduce_mean(truth)+tf.reduce_mean(pred)

djc_global_loss_list['dummy_loss']=dummy_loss