#! /bin/env python3

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('model')
parser.add_argument('output', help='output files name')
#parser.add_argument("", action='store_true')
parser.add_argument("--batch", type=int, default=1, help='batch size to be embedded in deployment')
args = parser.parse_args()


from keras.models import load_model
from keras import backend as K
from tensorflow.python.tools import optimize_for_inference_lib
from DeepJetCore.customObjects import get_custom_objects

custom_objs = get_custom_objects()

import tensorflow as tf
sess = tf.Session()
K.set_session(sess)



K.set_learning_phase(False) #FUNDAMENTAL! this MUST be before loading the model!
model=load_model(args.model, custom_objects=custom_objs)

output_names = [format_name(i.name) for i in model.outputs]
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names)

tf.train.write_graph(
   constant_graph, "", 
   args.output if args.output.endswith('.pb') else '%s.pb' % args.output,
   as_text=False
   )

txt_config = args.output.replace('.pb', '.config.pbtxt') \
   if args.output.endswith('.pb') else '%s.config.pbtxt' % args.output
with open(txt_config, 'w') as config:
   for feed in model.inputs:
      #the first element is always the batch size (None in the graph, needs to be defined here)
      shape = [args.batch] + [int(i) for i in feed.shape[1:]]
      shape = ['    dim { size : %s }' % i for i in shape]
      shape = '\n'.join(shape)
      config.write('''feed {
  id { node_name: "%s" }
  shape {
%s
  }
}
''' % (format_name(feed.name), shape))

   config.write('\n')
   for fetch in output_names:
      config.write('fetch {\n  id { node_name: "%s" }\n}\n' % fetch)
