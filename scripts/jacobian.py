import tensorflow as tf
import numpy as np 
input_names = ["input_{}:0".format(i) for i in range(1,6)]
output_names = ["ID_pred/Softmax:0", "regression_pred/BiasAdd:0"]


from DataCollection import DataCollection
testd=DataCollection()
testd.readFromFile('/eos/cms/store/cmst3/group/dehep/DeepJet/Predictions/Jan/DF_FT_fullRec_reg_BN/qcd_600_800_CFR/dataCollection.dc')
fullpath=testd.getSamplePath(testd.samples[0])
td=testd.dataclass
td.readIn(fullpath)
x=td.x
y=td.y
truth  = np.asarray(y[0])
np.save("truth",truth)
for input in range(len(x)-1): # last input is crap
  np.save("x_"+str(input),x[input])  

#np.save("inputLabels.npy",all_input)
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('testmodel/tf.meta')
  new_saver.restore(sess, 'testmodel/tf')
  
  x1 = tf.get_default_graph().get_tensor_by_name(input_names[0])
  x2 = tf.get_default_graph().get_tensor_by_name(input_names[1])
  x3 = tf.get_default_graph().get_tensor_by_name(input_names[2])
  x4 = tf.get_default_graph().get_tensor_by_name(input_names[3])
  x5 = tf.get_default_graph().get_tensor_by_name(input_names[4])
 # x6 = tf.get_default_graph().get_tensor_by_name(input_names[5])
  y_est = tf.get_default_graph().get_tensor_by_name(output_names[0]) 
  norm = tf.get_default_graph().get_tensor_by_name('batch_normalization_1/keras_learning_phase:0')

#  y_out = sess.run(y_est, feed_dict={x1: x[0] ,x2: x[1],x3: x[2],x4:x[3] ,x5:x[4],norm : False})
#  sess.run(feed_dict={y: }
  print (type(y_est), " ", y_est.shape)

  y_list_inputs = []
  print " length  " , len(y_list_inputs)
  # loop over the dirrefrent in puts
  for input in range(len(x)-1):
    y_list_inputs = []
    print " length  " , len(y_list_inputs), " ", input
    # loop over label, use only y[0], i.e. the classes
    for label in range (y[0].shape[1]):
      y_est_label = y_est[:,label:label+1] 
      y_list_inputs += [sess.run(tf.gradients(y_est_label, [x1,x2,x3,x4,x5])[input], feed_dict={x1: x[0] ,x2: x[1],x3: x[2],x4:x[3] ,x5:x[4],norm : False})]
    print " length  " , len(y_list_inputs)
    myarray = np.asarray(y_list_inputs)
    print ("shape", myarray.shape, " " , np.sum(myarray[0]) )
    print ("shape", myarray.shape, " " , np.sum(myarray[0]) )
    np.save("jacob_input"+str(input),myarray)
    jacob = np.load("jacob_input0.npy","r")
    print (jacob.shape)
    

  
#  y_list_Label = np.asarray(y_list_Label)
 # print "y_list_Label shape " ,  y_list_Label.shape
 # np.save("jacob.npy",y_list_Label)
 
  #np.save("inputLabels.npy",all_input)


def eval_session(inputs, outputs):
    outputs.update(sess.run(dict(zip(outputs.keys(), outputs.keys())), feed_dict=inputs))

