from keras import backend as K

from tensorflow import where, greater, abs, zeros_like, exp
import tensorflow as tf

global_loss_list={}

#whenever a new loss function is created, please add it to the global_loss_list dictionary!


def huberishLoss_noUnc(y_true, x_pred):
    
    
    dxrel=(x_pred - y_true)/1#(K.clip(K.abs(y_true+0.1),K.epsilon(),None))
    dxrel=K.clip(dxrel,-1e6,1e6)
    
    #defines the inverse of starting point of the linear behaviour
    scaler=2
    
    dxabs=K.abs(scaler* dxrel)
    dxsq=K.square(scaler * dxrel)
    dxp4=K.square(dxsq)
    
    lossval=dxsq / (1+dxp4) + (2*dxabs -1)/(1 + 1/dxp4)
    #K.clip(lossval,-1e6,1e6)
    
    return K.mean( lossval , axis=-1)
    


global_loss_list['huberishLoss_noUnc']=huberishLoss_noUnc



def loss_NLL(y_true, x):
    """
    This loss is the negative log likelyhood for gaussian pdf.
    See e.g. http://bayesiandeeplearning.org/papers/BDL_29.pdf for details
    Generally it might be better to even use Mixture density networks (i.e. more complex pdf than single gauss, see:
    https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
    """
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.log(K.square(x_sig))  + K.square(x_pred - y_true)/K.square(x_sig)/2.,    axis=-1)

#please always register the loss function here
global_loss_list['loss_NLL']=loss_NLL

def loss_meansquared(y_true, x):
    """
    This loss is a standard mean squared error loss with a dummy for the uncertainty, 
    which will just get minimised to 0.
    """
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.square(x_sig)  + K.square(x_pred - y_true)/2.,    axis=-1)

#please always register the loss function here
global_loss_list['loss_meansquared']=loss_meansquared


def loss_logcosh(y_true, x):
    """
    This loss implements a logcosh loss with a dummy for the uncertainty.
    It approximates a mean-squared loss for small differences and a linear one for
    large differences, therefore it is conceptually similar to the Huber loss.
    This loss here is scaled, such that it start becoming linear around 4-5 sigma
    """
    scalefactor_a=30
    scalefactor_b=0.4
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    def cosh(y):
        return (K.exp(y) + K.exp(-y)) / 2
    
    return K.mean(0.5*K.square(x_sig))   + K.mean(scalefactor_a* K.log(cosh( scalefactor_b*(x_pred - y_true))), axis=-1)
    


global_loss_list['loss_logcosh']=loss_logcosh


def loss_logcosh_noUnc(y_true, x_pred):
    """
    This loss implements a logcosh loss without a dummy for the uncertainty.
    It approximates a mean-squared loss for small differences and a linear one for
    large differences, therefore it is conceptually similar to the Huber loss.
    This loss here is scaled, such that it start becoming linear around 4-5 sigma
    """
    scalefactor_a=1.
    scalefactor_b=3.
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    dxrel=(x_pred - y_true)/(y_true+0.0001)
    def cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    
    return scalefactor_a*K.mean( K.log(cosh(scalefactor_b*dxrel)), axis=-1)
    


global_loss_list['loss_logcosh_noUnc']=loss_logcosh_noUnc

# The below is to use multiple gaussians for regression

## https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb
## these next three functions are from Axel Brando and open source, but credits need be to https://creativecommons.org/licenses/by-sa/4.0/ in case we use it

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), 
                       axis=axis, keepdims=True))+x_max
                       

global_loss_list['log_sum_exp']=log_sum_exp

def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    
    #Note: The output size will be (c + 2) * m = 6
    c = 1 #The number of outputs we want to predict
    m = 2 #The number of distributions we want to use in the mixture
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-8,1.))
    
    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
    - float(c) * K.log(sigma) \
    - K.sum((K.expand_dims(y_true,2) - mu)**2, axis=1)/(2*(sigma)**2)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res


global_loss_list['mean_log_Gaussian_like']=mean_log_Gaussian_like


def mean_log_LaPlace_like(y_true, parameters):
    """Mean Log Laplace Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    #Note: The output size will be (c + 2) * m = 6
    c = 1 #The number of outputs we want to predict
    m = 2 #The number of distributions we want to use in the mixture
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-2,1.))
    
    exponent = K.log(alpha) - float(c) * K.log(2 * sigma) \
    - K.sum(K.abs(K.expand_dims(y_true,2) - mu), axis=1)/(sigma)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

global_loss_list['mean_log_LaPlace_like']=mean_log_LaPlace_like


def moment_loss(y_true, y_pred):
	'''this methods calculates moments for different "bins", e.g. two; 1 data and 2 mc, and returns the difference between these moments of the different bins. The bins are passed in the true labels. This loss supports only one prediction output (the prediction is a single value per element)'''
	# The below counts the entries in the histogram vector, i.e. the actual mini batch size
	h_entr = K.sum(y_true,axis=0)
    
	## first moment ##
	# Multiply the histogram vectors with estimated propability y_pred
	h_fill = y_true * y_pred
    
	# Sum each histogram vector
	Sum =K.sum(h_fill,axis=0)
    
  # Devide sum by entries (batch size) (i.e. mean, first moment)
	Sum = Sum/h_entr
    
  # Devide per vector mean by average mean, i.e. now SUM is a vector of the relative deviations from the absolute mean
	Sum = Sum/K.mean(y_pred)
	
	## second moment, same logic as before, just squared
	y_pred2 = y_pred-K.mean(y_pred)
	h_fill2 = y_true * y_pred2*y_pred2
	Sum2 =K.sum(h_fill2,axis=0)
	Sum2 = Sum2/h_entr
	Sum2 = Sum2/K.mean(y_pred2*y_pred2)
	
	## third moment
	
	y_pred3 = y_pred-K.mean(y_pred)
	h_fill3 = y_true * y_pred3*y_pred3*y_pred3
	Sum3 =K.sum(h_fill3,axis=0)
	Sum3 = Sum3/h_entr
	Sum3 = Sum3/K.mean(y_pred2*y_pred2*y_pred2)
	
	## fourth moment
	
	y_pred4 = y_pred-K.mean(y_pred)
	h_fill4 = y_true * y_pred4*y_pred4*y_pred4*y_pred4
	Sum4 =K.sum(h_fill4,axis=0)
	Sum4 = Sum4/h_entr
	Sum4 = Sum4/K.mean(y_pred4*y_pred4*y_pred4*y_pred4)
	
	return  K.mean(K.square(Sum-1)) + K.mean(K.square(Sum2-1))  +  K.mean(K.square(Sum3-1))  + K.mean(K.square(Sum4-1))

global_loss_list['moment_loss'] = moment_loss



def nd_moment_loss(y_true, y_pred):
	'''Extension of the moment_loss to the case where the prediction in multi-dimensional. 
This methods calculates moments for different "bins", e.g. two; 1 data and 2 mc, and returns the difference between these moments of the different bins. The bins are passed in the true labels.'''
	# The below counts the entries in the histogram vector, i.e. the actual mini batch size
	h_entr = K.sum(y_true,axis=0)
	  
	## first moment ##
	# Multiply the histogram vectors with estimated propability y_pred
	# and sum each histogram vector
	#Rows: predition classes, Colums: bins
	Sum = tf.transpose(tf.matmul(
			tf.transpose(y_true), y_pred
			)) 
	  
	# Devide sum by entries (batch size) (i.e. mean, first moment)
	Sum /= h_entr
	
	# Devide per vector mean by average mean in each class, i.e. now SUM is a vector of the relative deviations from the absolute mean
	#Rows: bins, Columns: prediction classes
	Sum = tf.transpose(Sum) / K.mean(y_pred, axis=0)
	
	#higer moments: common var
	y_pred_deviation = y_pred - K.mean(y_pred, axis=0)
	
	## second moment, same logic as before, just squared  
	#Rows: predition classes, Colums: bins
	Sum2 = tf.transpose(tf.matmul(
			tf.transpose(y_true), y_pred_deviation**2
			))
	Sum2 /= h_entr
	Sum2 = tf.transpose(Sum2)/K.mean(y_pred_deviation**2, axis=0)
	
	## third moment
	Sum3 = tf.transpose(tf.matmul(
			tf.transpose(y_true), y_pred_deviation**3
			))
	Sum3 /= h_entr
	Sum3 = tf.transpose(Sum3)/K.mean(y_pred_deviation**3, axis=0)
	
	## fourth moment
	Sum4 = tf.transpose(tf.matmul(
			tf.transpose(y_true), y_pred_deviation**4
			))
	Sum4 /= h_entr
	Sum4 = tf.transpose(Sum4)/K.mean(y_pred_deviation**4, axis=0)
	
	return  K.mean(K.square(Sum-1)) + K.mean(K.square(Sum2-1))  +  K.mean(K.square(Sum3-1))  + K.mean(K.square(Sum4-1))

global_loss_list['nd_moment_loss'] = nd_moment_loss

def nd_moment_factory(nmoments):
	if not isinstance(nmoments, int) and nmoments < 1:
		raise ValueError('The number of moments used must be integer and > 1')
	def nd_moments_(y_true, y_pred):
		# The below counts the entries in the histogram vector, i.e. the actual mini batch size
		h_entr = K.sum(y_true,axis=0)
		  
		## first moment, it's always there ##
		# Multiply the histogram vectors with estimated propability y_pred
		# and sum each histogram vector
		#Rows: predition classes, Colums: bins
		Sum = tf.transpose(tf.matmul(
				tf.transpose(y_true), y_pred
				)) 
		  
		# Devide sum by entries (batch size) (i.e. mean, first moment)
		Sum /= h_entr
		
		# Devide per vector mean by average mean in each class, i.e. now SUM is a vector of the relative deviations from the absolute mean
		#Rows: bins, Columns: prediction classes
		Sum = tf.transpose(Sum) / K.mean(y_pred, axis=0)
		
		#higer moments: common var
		y_pred_deviation = y_pred - K.mean(y_pred, axis=0)
		nsums = [K.mean(K.square(Sum-1))]
		for idx in range(2, nmoments+1): #from 2 to N (included)
			isum = tf.transpose(tf.matmul(
					tf.transpose(y_true), y_pred_deviation**idx
					))
			isum /= h_entr
			isum = tf.transpose(isum)/K.mean(y_pred_deviation**idx, axis=0)
			nsums.append(K.mean(K.square(isum-1)))
		return tf.add_n(nsums)
	return nd_moments_
		
for i in range(1, 5):
	global_loss_list['nd_%dmoment_loss' % i] = nd_moment_factory(i)
