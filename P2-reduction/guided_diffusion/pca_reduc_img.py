import numpy as np
from PIL import Image


#reduc = np.load('../data/obama/reduc.npy')

#for i in range(200):
#    np.save('../data/obama/PCA_100_flipadd/PCA_{}.npy'.format(i), reduc[i])


reduc = np.load('../data/obama/reduc.npy')
print(reduc[0])
print(reduc[1])
print(reduc[2])
'''
comp = np.load('../data/obama/comp.npy')
mean = np.load('../data/obama/mean.npy')
print(reduc[0].ndim)
print(comp.ndim)
print(mean.ndim)
restore = reduc[0] @ comp + mean 
restore = (restore+1)*127.5
print(restore.reshape(1,3,256,256))
'''

