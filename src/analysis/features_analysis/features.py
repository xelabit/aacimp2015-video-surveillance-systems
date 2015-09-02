import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from scipy.misc import imread, imsave, imresize

def imgs2gray(imgs):
	return map(lambda x : cv2.cvtColor(x,cv2.COLOR_BGR2GRAY), imgs)

def imgs2corners(imgs, num_corners):
	return map(lambda x : np.int0(cv2.goodFeaturesToTrack(x, num_corners, 0.01, 10)), imgs)

def plotCorners(img, corners):
	plt.figure(figsize = (10,10))
	
	for i in corners:
		x, y = i.ravel()
		cv2.circle(img, (x,y), 3, (255,0,0))

	plt.imshow(img)

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def corners_bin(corn, bin_sizes):
    corn = corn[:,0,:]
    st_feat = np.zeros((240, 320), dtype=np.int) # BAD CODE
    for i in corn:
        st_feat[i[1],i[0]] = 1
    bins = rebin(st_feat, bin_sizes)    
    return bins.astype(int)    

def list2array(Xlist):
	Xarr = np.empty([2087, 100, 1, 2]) # BAD CODE

	for i in xrange(len(Xlist[0])):
		Xarr[i] = Xlist[i] 

	return Xarr