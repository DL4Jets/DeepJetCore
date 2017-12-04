import matplotlib.pyplot as plt
import matplotlib
import numpy as np
jacob = np.load("jacob_input0.npy")
print (jacob.shape)

jacobians = []
inputs = []

for i in range(4):
    name = 'plots/jacob_input'+str(i)+'.npy'
    jacobians += [np.load(name)]
    inputs += [np.load('plots/x_'+str(i)+'.npy')]


print ('loaded')
import time

def slideshow(jacobians,inputs,title,globalNames):
    axis_l =(np.asarray(range(0,51) ) -25)/5.
    axis_f =(np.asarray(range(0,51) ) -25)/100.
    means = []
    rms  = []
    for l in range (1):
        for i in range(18): #loops over feture of global variables
            plt.figure(i+l*15,figsize=(15, 5))
            plt.subplot(131)
            plt.ylabel('entries')
            plt.title(title)
            xlabel = 'partial derivative: '+ globalNames[i] + ' label_' +str(l)
            plt.xlabel(xlabel)
            thisjabob  = jacobians
            thisjabob = thisjabob[l,:,i]
            print (xlabel , " mean ", np.mean(thisjabob), " rms " ,  np.sqrt(np.mean(thisjabob**2)))
            print ("feature mean ", np.mean(inputs[:,i]), " rms " ,  np.sqrt(np.mean((inputs[:,i]-np.mean(inputs[:,i]))**2)))
            means+=[np.mean(thisjabob)]
            rms+=[np.sqrt(np.mean((thisjabob-np.mean(thisjabob))**2))]
            #plt.text(100, .0, 'mean: '+str(np.mean(thisjabob))+' rms: '+str(np.sqrt(np.mean(thisjabob**2))))
            plt.hist(thisjabob,bins = axis_f)
            x = inputs
            #plt.hist(x[:,i],bins = 50)
            plt.draw()
            plt.subplot(132)
            plt.hist(x[:,i],bins = axis_l)
            xlabel = globalNames[i]
            plt.xlabel(xlabel)
            plt.draw()
            plt.subplot(133)
            plt.title(title)
            #plt.title('global features')
            plt.ylabel('partial derivative [a.u.]')
            xlabel = globalNames[i]
            plt.xlabel(xlabel)
#x = inputs
            print ((np.asarray(range(0,51) ) -25)/10.  )
            axis2d =(np.asarray(range(0,51) ) -25)/10.
            a,_,_ = np.histogram2d(thisjabob,x[:,i], bins = [ axis_f, axis_l] )
            plt.imshow(a, cmap='hot', interpolation='nearest',norm=matplotlib.colors.LogNorm())
            #plt.scatter(thisjabob,x[:,i], cmap='hot', interpolation='nearest',norm=matplotlib.colors.LogNorm())
            
            plt.draw()
            plt.pause(0.001)
            input("Press [enter] to continue.")

    return (np.asarray(means),np.asarray(rms))


ccands_ = [
           'Cpfcan_BtagPf_trackEtaRel',
           'Cpfcan_BtagPf_trackPtRel',
           'Cpfcan_BtagPf_trackPPar',
           'Cpfcan_BtagPf_trackDeltaR',
           'Cpfcan_BtagPf_trackPParRatio',
           'Cpfcan_BtagPf_trackSip2dVal',
           'Cpfcan_BtagPf_trackSip2dSig',
           'Cpfcan_BtagPf_trackSip3dVal',
           'Cpfcan_BtagPf_trackSip3dSig',
           'Cpfcan_BtagPf_trackJetDistVal',
           'Cpfcan_BtagPf_trackJetDistSig',
           'Cpfcan_ptrel',
           'Cpfcan_drminsv',
           'Cpfcan_fromPV',
           'Cpfcan_VTX_ass',
           'Cpfcan_puppiw',
           'Cpfcan_chi2',
           'Cpfcan_quality'
]

globalNames_ = ['jet_pt', 'jet_eta',
 'nCpfcand','nNpfcand',
 'nsv','npv',
 'TagVarCSV_trackSumJetEtRatio',
 'TagVarCSV_trackSumJetDeltaR',
 'TagVarCSV_vertexCategory',
 'TagVarCSV_trackSip2dValAboveCharm',
 'TagVarCSV_trackSip2dSigAboveCharm',
 'TagVarCSV_trackSip3dValAboveCharm',
 'TagVarCSV_trackSip3dSigAboveCharm',
 'TagVarCSV_jetNSelectedTracks',
 'TagVarCSV_jetNTracksEtaRel']

print("len ", len(ccands_))

#slideshow( jacobians[0],inputs[0],'global features',globalNames)
print(jacobians[1].shape  , " ", jacobians[1][:,:,0,:].shape)
print(inputs[1].shape  , " ", inputs[1][:,0,:].shape)

(means,rms) = slideshow( jacobians[1][:,:,0,:],inputs[1][:,0,:],'global features',ccands_)

print(means)
plt.figure(1000)
plt.subplot(121)
plt.plot(means, 'ro')
plt.xlabel('index of feature')
plt.ylabel('mean')

plt.draw()

plt.subplot(122)

plt.plot(rms, 'ro')
plt.xlabel('index of feature')
plt.ylabel('rms')
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")



print ('done')


