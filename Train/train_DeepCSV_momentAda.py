from training_base import training_base
from Losses import nd_moment_loss
from MultiDataCollection import MultiDataCollection
from pdb import set_trace

#also does all the parsing
train=training_base(testrun=False, collection_class=MultiDataCollection)
print 'Inited'
sizes = train.train_data.sizes
norm = float(sizes[2])/sizes[1] #normalization because samples have different sizes
train.train_data.setFlags([[1,0], [0,norm], [0,1]])
train.train_data.addYs([[[0,0]], [[1,0]], [[0,1]]])

evt = train.train_data.generator().next()
set_trace()

train.val_data.setFlags([[1,0], [0,norm], [0,1]])
train.val_data.addYs([[[0,0]], [[1,0]], [[0,1]]])

if not train.modelSet():
    from models import dense_model_moments
    print 'Setting model'
    train.setModel(dense_model_moments, dropoutRate=0.1)
    
    train.compileModel(
			learningrate=0.003,
			loss=['categorical_crossentropy', nd_moment_loss],
			#loss_weights=[1., 0.000000000001],
			metrics=['accuracy'],
		)


model,history = train.trainModel(
	nepochs=50, 
	batchsize=5000, 
	stop_patience=300, 
	lr_factor=0.5, 
	lr_patience=10, 
	lr_epsilon=0.0001, 
	lr_cooldown=2, 
	lr_minimum=0.0001, 
	maxqsize=100
)
