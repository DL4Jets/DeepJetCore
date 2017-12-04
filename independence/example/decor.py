# Simple example to present a methods to add a measure of independence w.r.t. a variable "a" to the loss.


import numpy as np

a=np.random.rand(10000)*5
b=np.random.rand(10000)*5
c=b*b

# OH will be the truth "y" input to the network
# OH contains both, the actual truth per sample and the actual bin (one hot encoded) of the variable to be independent of
OH = np.zeros((10000,6))
for i in range (0,10000):
    # bin of a (want to be independent of a)
    OH[i,int(a[i])]=1
    # aimed truth (target) (want to esimate b*b+a)
    OH[i,5] = (b*b+a) [i]
#print ('OH,a')
#print (OH)
#print (a)


import matplotlib.pyplot as plt

x = np.vstack((a,c)).T
## WARNING
# y = np.vstack((c,a)).T
#print (y.shape, ' ',x.shape)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# not used
# this decorrelates, yet that is typically not enough in real use case
def loss_decorr(y_in,x):
    print (y_in.shape, ' ',x.shape)
    y = y_in[:,1:]
    c = y_in[:,:1]
    print (y.shape, ' ',a.shape)
    return K.mean((K.mean(y)-y)*(K.mean(x)-x))*K.mean((K.mean(y)-y)*(K.mean(x)-x)) +0.5*K.mean(K.square(c - x))

# not used, alternative independence test
# sorting too slow, not deriveable quickly, not supported in KERAS, likely not on option for DNN loss where performance matters
def loss_indep(y,x):
    y_sorted = np.sort(x)
    y_sorted = sorted (x, key=lambda a_entry: a_entry[1])[0]
    #https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0697-7
    y_diff = y_sorted[:-1]-y_sorted[1:]
    W_obs = numpy.mean(y_diff*y_diff)
    y_diff_ran = x_numpy[:-1] - x_numpy[1:]
    y_true = y_diff_ran>W_obs
    value = numpy.sum(y_true)/y_diff_ran.size()
    return value

h = OH[:,0:5]
y = OH[:,5:]
h_entr = np.sum(h,axis=0)
print ('sum', h_entr)


# New idea for DNN loss, use moments (or other analytical functions) and histograms. Both easy from computational point of view
# this loss can be used in KERAS and uses the Keras backend "K", i.e. derivatives ect. already included
def loss_moment(y_in,x):
    
    # h is the histogram vectore "one hot encoded" (5 bins in this case), techically part of the "truth" y
    h = y_in[:,0:5]
    y = y_in[:,5:]
    
    # The below counts the entries in the histogram vector
    h_entr = K.sum(h,axis=0)
    
    ## first moment ##
    
    # Multiply the histogram vectors with estimated propability x
    h_fill = h * x
    
    # Sum each histogram vector
    Sum =K.sum(h_fill,axis=0)
    
    # Devide sum by entries (i.e. mean, first moment)
    Sum = Sum/h_entr
    
    # Devide per vector mean by average mean
    Sum = Sum/K.mean(x)
    
    ## second moment
    
    x2 = x-K.mean(x)
    h_fill2 = h * x2*x2
    Sum2 =K.sum(h_fill2,axis=0)
    Sum2 = Sum2/h_entr
    Sum2 = Sum2/K.mean(x2*x2)
    
    ## the loss, sum RMS + two moments (or more). The RMS is downweighted.
    
    return  0.005*K.mean(K.square(y - x)) + K.mean(K.square(Sum-1)) + K.mean(K.square(Sum2-1))

## debug loss in numpy to test keras loss. Can not be used in KERAS, yet allows to debug KERAS using more precoded methods of numpy scikit ect.
import scipy
from scipy import stats
def loss_moment_numpy(y_in,x):
    
    h = y_in[:,0:5]
    y = y_in[:,5:]
    h_entr = np.sum(h,axis=0)
    #print ('shape', h.shape)
    h_fill = h*np.reshape(x, (10000,1))
    #print (h_entr.shape, ' ',h_fill.shape)
    Sum =np.sum(h_fill,axis=0)
    #print(stats.moment(h_fill,axis=0), ' and second ', scipy.stats.moment(h_fill,moment=2,axis=0))
    Sum = Sum/h_entr
    print ('Means: ', Sum)
    Sum = Sum/np.mean(x)
    x2 = x-np.mean(x)
    #print ('x2',x2)
    h_fill2 = h*np.reshape(x2*x2,(10000,1))
    #  print (h_entr.shape, ' ',Sum.shape)
    #  print ('x2',h_fill2)
    Sum2 =np.sum(h_fill2,axis=0)
    # print(stats.moment(h_fill), ' and second ', scipy.stats.moment(h_fill,2))
    Sum2 = Sum2/h_entr
    print('var : ', Sum2, ' ',np.mean(x2*x2))
    Sum2 = Sum2/np.mean(x2*x2)
    print ('RMS ', np.mean(np.square(y - x)))
    #    print ('Moments: ', Sum,' ' ,Sum2)
    print (' Loss',  0.005*np.mean(np.square(y - x)),' + ', np.mean(np.square(Sum-1)), ' + ',np.mean(np.square(Sum2-1))  )
    # print ('Av moment 2 ',np.mean(np.square(Sum2-np.mean(x2*x2))))
    # print (Sum, 'moments ',np.mean(x), ' ',np.square(Sum-np.mean(x)))
    #print ('Av moment 1 ',np.mean(np.square(Sum-np.mean(x))))
    #print (' err: ', np.mean(np.square(y - x)))
    print ( np.mean(np.square(y - x)) +np.mean(np.square(Sum-1))  + np.mean(np.square(Sum2-1)))
    return  np.mean(np.square(y - x))



# make a simple model:

model = Sequential()
model.add(Dense(1, activation='linear', input_shape=(2,)))
model.summary()
model.compile(loss=loss_moment,optimizer='adam' )

# batch size is huge because of need to evaluate independence
model.fit(x,OH, batch_size=40000, nb_epoch=20000, verbose=1,validation_split=0.3)
# get the truth:
ouput = model.predict(x)


# some printouts

print (a,ouput)
res = np.hstack((a.ravel(),b.ravel()))
print('custom loss b*b+5')
loss_moment_numpy(OH,ouput)
myest = b*b+2.5
print (' my guess ', myest , ' fit ' , ouput )
loss_moment_numpy(OH,myest)


# some plots:

truth =(b*b+a).ravel()
heatmap, xedges, yedges = np.histogram2d(truth,ouput.ravel(), bins=50)
#print(heatmap)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()
#fig = plt.subplots()
#cbar = fig.colorbar(cax, ticks=[-1000, 0, 1000])
difft=truth-ouput.ravel()
heatmap, xedges, yedges = np.histogram2d(a.ravel(),ouput.ravel(), bins=[5,200])
#print(heatmap)
#for i in range (0,5):
#    print ('mean ', heatmap[:][i:i+1], ' ',np.mean(heatmap[:][i:i+1]))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.ylim((0,100))
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()
#diff =(b*b).ravel() - ouput.ravel()
myhist, bins, patches = plt.hist(difft, 50)
#plt.plot(a)
plt.show()

## The result was that it in principle worked. It is clear that the loss needs carefull weighting of it elements. I.e. the RMS starts >>> greater than the moment independence test. This however only requires to maybe use an NLL instead of RMS and for x-entropy it should work as well.



