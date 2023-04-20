# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:08:22 2023

@author: John Breyner Tacan
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms





img_rgb=cv2.imread("img_Hematoxilina_Referencia.png")


img_rgb_2=img_rgb.reshape((-1,3))

img_rgb_2=np.float32(img_rgb_2)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#cluster
k=2
attempts=2

ret,label,center=cv2.kmeans(img_rgb_2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)


center=np.uint8(center)

res=center[label.flatten()]
res2=res.reshape((img_rgb.shape))

gray_img=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

cv2.imshow("img_rgb_referencia",img_rgb)
cv2.imshow("gray_img",gray_img)
cv2.imshow("res2",res2)
cv2.waitKey(60000)
cv2.destroyAllWindows() 







