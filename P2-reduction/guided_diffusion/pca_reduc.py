import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image



all_imgs = []
for i in range(200):
    img = Image.open('../data/obama/100_flipadd/{}.jpg'.format(i))
    img = img.convert('RGB')
    img = np.array(img)
    img = img/127.5 -1
    img_flat = img.flatten()
    all_imgs.append(img_flat)

all_imgs = np.array(all_imgs)



def pca_reduction(imgs):
    pca = PCA(n_components = 200)

    pca.fit(imgs)
    trans_pca = pca.transform(imgs)
    #print(trans_pca.shape)

    return trans_pca, pca.components_, pca.mean_


def pca_restore(reduc, components, mean):
    restore = reduc @ components + mean    

    return restore


reduc, comp, mean = pca_reduction(all_imgs)
np.save('../data/obama/reduc', reduc)
np.save('../data/obama/comp', comp)
np.save('../data/obama/mean', mean)

#final = pca_restore(reduc, comp, mean)

#diff = (all_imgs+1)*127.5 - final
#print(abs(diff).max())



'''
def pca(img):
    I = np.array(img)    

    pca_r = PCA(n_components=50)
    pca_g = PCA(n_components=50)
    pca_b = PCA(n_components=50)

    pca_r.fit(I[:,:,0])
    trans_pca_r = pca_r.transform(I[:,:,0])
    pca_g.fit(I[:,:,1])
    trans_pca_g = pca_g.transform(I[:,:,1])
    pca_b.fit(I[:,:,2])
    trans_pca_b = pca_b.transform(I[:,:,2])

    r_restore = pca_r.inverse_transform(trans_pca_r)
    g_restore = pca_g.inverse_transform(trans_pca_g)
    b_restore = pca_b.inverse_transform(trans_pca_b)
    
    restore = np.stack((r_restore,g_restore,b_restore), axis = 2)
    final = (restore+1)*127.5

    return final


img = cv2.imread('../data/obama/100/0.jpg')
img_norm = img/127.5 -1
final = pca(img_norm)
cv2.imwrite('samples/pca_final_orig.jpg', final)
'''