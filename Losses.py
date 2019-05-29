
djc_global_loss_list = {}

import tensorflow as tf

def null_loss(truth, pred):
    return tf.reduce_mean(truth*0.)+tf.reduce_mean(pred*0.)

djc_global_loss_list['null_loss']=null_loss