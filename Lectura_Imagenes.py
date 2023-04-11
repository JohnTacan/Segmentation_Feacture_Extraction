"""
Created on Thu Apr  6 11:29:58 2023

@author: John Breyner Tacan
"""
import cv2 
import glob
import numpy as np
from skimage.color import rgb2lab,lab2rgb
from skimage import data
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms



def H_E(img_matched):
    #Hematoxilina & Eosina

    img=cv2.cvtColor(img_matched,cv2.COLOR_BGR2RGB)


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



def HED(img_matched):
    
    import numpy as np
    from skimage.color import rgb2hed, hed2rgb

    # Example IHC image
    ihc_rgb = img_rgb
    ihc_hed = rgb2hed(ihc_rgb)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    
    
    return ihc_h














path_b="C:/Trabajo_de_Grado/Segmentation_Feacture_Extraction/Segmentation_Feacture_Extraction/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/Benign/Prueba_b/*.png"
#path_m="C:/Trabajo_de_Grado/Segmentation_Feacture_Extraction/Segmentation_Feacture_Extraction/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/Invasive/Prueba_m/*.tif"
for file in glob.glob(path_b):
    
    img_rgb=cv2.imread(file)
    
    reference=cv2.imread("reference.png")
    
    #Calculo del histograma 
    """
    channels=cv2.split(img_rgb)
    
    colors=('b','g','r')
    
    for(channel,color) in zip(channels,colors):
        
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color )
        
    cv2.imshow("img",img_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()   
        
        

    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    
    """
    #img_rgb[:,:,0] = cv2.equalizeHist(img_rgb[:,:,0])
    #img_rgb[:,:,1] = cv2.equalizeHist(img_rgb[:,:,1])
    #img_rgb[:,:,2] = cv2.equalizeHist(img_rgb[:,:,2])
    
    
    #CLAHE
    img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB)
    
    #img_rgb= cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))

    img_lab[:,:,0] = clahe.apply( img_lab[:,:,0])
    
    img_rgb_clahe= cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    
    #Matching histogram
    
    img_matched = match_histograms(img_rgb_clahe, reference ,
                           multichannel=True)
    
    img_hematoxilina=H_E(img_matched)
    
    img_hematoxilina2=HED(img_matched)
    
    
    cv2.imshow("img",img_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    
    cv2.imshow("img_matched",img_matched)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    
    cv2.imshow("img_Hematoxilina",img_hematoxilina)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    
    cv2.imshow("img_Hematoxilina",img_hematoxilina2)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    
    
    
    
    
    
    """
    #Calculo del histograma 
    channels=cv2.split( img_rgb)
    
    for(channel,color) in zip(channels,colors):
        
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color )
        

    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    
    cv2.imshow("img_matched",  matched)
    #cv2.imshow("img",  img_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()   
    
    
    """
    
    
    
    
    
    
    """
    
    #Distribuir la imagen en parches 
    PATCH_SIZE=512
    
    #cell_locations (y,x)
    cell_locations=[(0,0),(0,512),(0,1024),(0,1536),
                    (512,0),(512,512),(512,1024),(512,1536),
                    (1024,0),(1024,512),(1024,1024),(1024,1536),
                    ]
    
    cell_patches=[]

    for loc in cell_locations:
        cell_patches.append(img_rgb[loc[0]:loc[0]+PATCH_SIZE,
                                    loc[1]:loc[1]+PATCH_SIZE])
    
    
    for patch in cell_patches:
        
        img_rgb_path=patch
        
        
        
        #(ihc_h,h,zdh)=HED(img_rgb_path)
        
        #cv2.imshow("img",ihc_d)
        #cv2.waitKey(4000)
        #cv2.destroyAllWindows()
        
        # Display
        #fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
        #ax = axes.ravel()    
    
        #ax[1].imshow(h)
        #ax[1].set_title("Hematoxylin")
        
    # Example IHC image
    #img_rgb2 = data.immunohistochemistry()
    
    #(ihc_d,h,zdh)=HED(img_rgb) 
    # Display
    #fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)
    #ax = axes.ravel()    

    #ax[1].imshow(ihc_d)
    #ax[1].set_title("Hematoxylin")       
       
        
        #fig = plt.figure()
        #axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
        #axis.imshow(zdh)
        #axis.set_title('Stain-separated image (rescaled)')
        #axis.axis('off')
        #plt.show()

       
    #cv2.imshow("img",ihc_d)
    #cv2.waitKey(60000)
    #cv2.destroyAllWindows()
        
        
    """
    
    
  
    