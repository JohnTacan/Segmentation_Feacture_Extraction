# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:08:22 2023

@author: John Breyner Tacan
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms



def H_E(img_rgb):
    #Hematoxilina & Eosina

    img=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)


    Io=240
    alpha=1
    beta=0.15

    ## Reference H&E OD matrix
    HERef= np.array([[0.5626,0.2159],
                   [0.7201,0.8012],
                   [0.4062,0.5581]])

    # Reference maximum stain concentrations for H&E

    maxCRef=np.array([1.9705,1.0308])

    #extract the height, width and num of channels of image

    h,w,c=img.shape

    #reshape image to multiple rows and 3 columns
    #Num of rows depends on the image size(wxh)

    img=img.reshape((-1,3))

    #calculate optical density
    #OD=-log10(I)
    #img.astype(np.float)

    OD=-np.log10((img.astype(float)+1)/Io)

    ODhat=OD[~np.any(OD<beta,axis=1)] # Returns an array whre OD values 


    eigvals,eigvecs=np.linalg.eigh(np.cov(ODhat.T))


    that=ODhat.dot(eigvecs[:,1:3])

    phi=np.arctan2(that[:,1],that[:,0])

    minPhi=np.percentile(phi,alpha)
    maxPhi=np.percentile(phi,100-alpha)


    vMin=eigvecs[:,1:3].dot(np.array([(np.cos(minPhi),np.sin(minPhi))]).T)
    vMax=eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi),np.sin(maxPhi))]).T)


    if vMin[0]>vMax[0]:
        HE=np.array((vMin[:,0],vMax[:,0])).T

    else:
        HE=np.array((vMax[:,0],vMin[:,0])).T


    Y=np.reshape(OD,(-1,3)).T

    C=np.linalg.lstsq(HE,Y,rcond=None)[0]

    maxC=np.array([np.percentile(C[0,:],99),np.percentile(C[1,:],99)])
    tmp=np.divide(maxC,maxCRef)
    C2=np.divide(C,tmp[:,np.newaxis])


    H=np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0],axis=1).dot(np.expand_dims(C2[0,:],axis=0))))
    H[H>255]=254
    H=np.reshape(H.T,(h,w,3)).astype(np.uint8)
    
            
    return H



img_rgb_referencia = np.zeros((512,512,3),np.uint8)

for i in range(0, 256):
    for j in range(0, 256): 
        img_rgb_referencia[i, j] = (131, 25 , 94)
        
for i in range(256, 512):
    for j in range(0, 256): 
        img_rgb_referencia[i, j] = (131, 25 , 94)
          
for i in range(0,256):
    for j in range(256, 512): 
        img_rgb_referencia[i, j] = (142, 76 , 255)
        
for i in range(256,512):
    for j in range(256, 512): 
        img_rgb_referencia[i, j] = (142, 76 , 255)



  


img_rgb=cv2.imread("SOB_M_DC-14-16448-400-004.png")


channels=cv2.split(img_rgb_referencia)

colors=('b','g','r')

for(channel,color) in zip(channels,colors):
    
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color )
    

plt.xlabel('intensidad de iluminacion')
plt.ylabel('cantidad de pixeles')
plt.show()


#CLAHE
img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB)



clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))

img_lab[:,:,0] = clahe.apply( img_lab[:,:,0])

img_rgb_clahe= cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


"""
#Calculo del histograma 
channels=cv2.split(img_rgb_clahe)

colors=('b','g','r')

for(channel,color) in zip(channels,colors):
    
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color )
    

plt.xlabel('intensidad de iluminacion_clahe')
plt.ylabel('cantidad de pixeles')
plt.show()
""" 



img_hematoxilina=H_E(img_rgb_referencia)


#Calculo del histograma 
channels=cv2.split(img_hematoxilina)

colors=('b','g','r')

for(channel,color) in zip(channels,colors):
    
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color )
    

plt.xlabel('intensidad de iluminacion_clahe')
plt.ylabel('cantidad de pixeles')
plt.show()



#cv2.imshow("img",img_rgb)
#cv2.imshow("img:clahe",img_rgb_clahe)
cv2.imshow("img_h",img_hematoxilina)
cv2.imshow("img_rgb_referencia",img_rgb_referencia)

cv2.waitKey(10000)
#cv2.destroyAllWindows() 







