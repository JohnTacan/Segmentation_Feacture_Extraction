"""
Created on Thu Apr  6 11:29:58 2023

@author: John Breyner Tacan
"""
import cv2 
import glob
import numpy as np
from skimage.color import rgb2lab,lab2rgb
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.segmentation import clear_border
import scipy.fftpack as fft
from skimage import measure, color, io,img_as_ubyte


    

def Watershed(img_dist_transform,img_hematoxilina,img_rgb):
    
    #img_gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    
    #Imagen_Filtrada_rgb=cv2.cvtColor(img_filtrada,cv2.COLOR_GRAY2RGB)
     
    kernel=np.ones((3,3),np.uint8)
    sure_bg=cv2.dilate(img_dist_transform,kernel,iterations=2)
    
    #sure_fg=cv2.dilate(img_thresholded,kernel,iterations=2)
    
    sure_bg = np.uint8(sure_bg)
    
    sure_fg=img_dist_transform
    
    sure_fg = np.uint8(sure_fg)
    
    
    unknown=cv2.subtract(sure_bg,sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    ret3, markers=cv2.connectedComponents(sure_fg)
    
    #markers=markers+10
    
    markers[unknown==255]=0
    
    
    markers=cv2.watershed(img_hematoxilina,markers)
    
    img_rgb[markers==-1]=[0,255,255]
    
    imagen_watershed=color.label2rgb(markers,bg_label=0)
    
    return imagen_watershed,markers,img_rgb


def Thresholded(img_blurred):
    
    ret, thresholded = cv2.threshold(img_blurred, 150, 255, cv2.THRESH_BINARY_INV)
    #thresholded=cv2.adaptiveThreshold(img_filtrada,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10)
    Clear_border=clear_border(thresholded)
    
    
    sz = 2
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*sz-1, 2*sz-1))
    
    
    #opening=cv2.morphologyEx(Clear_border,cv2.MORPH_OPEN,kernel,iterations=1)
    
    Imagen_Segmentada=cv2.morphologyEx(Clear_border,cv2.MORPH_CLOSE,kernel,iterations=5)
    
    
    
    #Imagen_Segmentada=Clear_border
    
    
    return Imagen_Segmentada


def Filtro_Paso_Alto(img_blurred):
    
    # Creamos un filtro pasa altas gaussiano de dimensiones 512x512
    F1=np.arange(-256,256,1)
    F2=np.arange(-256,256,1)
    [X,Y]=np.meshgrid(F1,F2)
    R=np.sqrt(X**2+Y**2)
    R=R/np.max(R)

    #sigma=0.009
    sigma = 0.009

    #Filtro pasa altas con función gaussiana
    Filt_Im = 1-np.exp(-(R**2)/(2*sigma**2))
    

    gray_image = cv2.resize(img_blurred,(512,512))

    gray_f=np.float64(gray_image)
    Fimg=fft.fft2(gray_f)
    Fsh_Image=fft.fftshift(Fimg)
    FFt_filtered=Fsh_Image*Filt_Im
    ImageFiltered = fft.ifft2(fft.ifftshift(FFt_filtered))
    ImageFilteredN = cv2.normalize(abs(ImageFiltered), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

    Imagen_filtrada = cv2.resize(ImageFilteredN,(700,460))
    
    
    return Imagen_filtrada


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





#path_b="C:/Trabajo_de_Grado/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/Imagenes_Muestra_B/*.png"
path_m="C:/Trabajo_de_Grado/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/Imagenes_Muestra/Imagenes_Muestra_B/*.png"

for file in glob.glob(path_m):
    
    img_rgb=cv2.imread(file)
    
  
    reference=cv2.imread("SOB_M_DC-14-16448-400-015.png")
    
    """
    #Calculo del histograma 
    
    channels=cv2.split(img_rgb)
    
    colors=('b','g','r')
    
    for(channel,color) in zip(channels,colors):
        
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color )
        
   
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    """
    """
    #Calculo del histograma 
    
    channels=cv2.split(reference)
    
    colors=('b','g','r')
    
    for(channel,color) in zip(channels,colors):
        
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color )
        
   
    plt.xlabel('intensidad de iluminacion_reference')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    
    """
    
    
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

    
    
    #Matching histogram
    
    img_matched = match_histograms(img_rgb_clahe, reference ,
                           multichannel=True)
    
    
    """
    #Calculo del histograma 
    channels=cv2.split(img_matched)
    
    colors=('b','g','r')
    
    for(channel,color) in zip(channels,colors):
        
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color )
        

    plt.xlabel('intensidad de iluminacion_matched')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    """
    
    #imagen hematoxilina
    img_hematoxilina=H_E(img_matched)
     
    
    #Imagen a escala de grices 
    
    #img_gray=cv2.cvtColor(img_hematoxilina,cv2.COLOR_BGR2GRAY)
    
    

    img_hematoxilina_2=img_hematoxilina.reshape((-1,3))
    
    img_hematoxilina_2=np.float32(img_hematoxilina_2)
    
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    
    #cluster
    k=2
    attempts=5
    
    ret,label,center=cv2.kmeans(img_hematoxilina_2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    
    
    center=np.uint8(center)
    
    res=center[label.flatten()]
    
    img_hematoxilina_kmeans=res.reshape((img_hematoxilina.shape))
    
    img_gray=cv2.cvtColor(img_hematoxilina_kmeans,cv2.COLOR_BGR2GRAY)
    
    
    
    # Imagen blurred 
    
    img_blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
    
    
    
    
    
    #Filtro paso alto
    img_filtrada=Filtro_Paso_Alto(img_blurred)
    
    #Thresholded
    img_thresholded=Thresholded(img_blurred)
    
    
    dist_transform = cv2.distanceTransform(img_thresholded,cv2.DIST_L2,5)
    ret2, img_dist_transform= cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    
    
    
    #Watershed
    img_watershed,markers_w,img_rgb_w=Watershed(img_dist_transform,img_hematoxilina,img_rgb)
    
    

    cv2.imshow("img_watershed",img_rgb_w)
    
    #cv2.imshow("img_thresholded",img_thresholded)
    #cv2.imshow("img_dist_transform",img_dist_transform)
    
    #cv2.imshow("img_hematoxilina",img_hematoxilina)
    #cv2.imshow("img_hematoxilina_kmeans",img_hematoxilina_kmeans)
    #cv2.imshow("img_gray",img_gray)
    cv2.waitKey(2000)
    cv2.destroyAllWindows() 

    
    """
    hist = cv2.calcHist([img_filtrada], [0], None, [256], [0, 256])
    plt.plot(hist, color='gray' )
        

    plt.xlabel('intensidad de iluminacion_matched')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    """
    

    """
    cv2.imshow("img",img_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    """
    
    """
    cv2.imshow("img_clahe",img_rgb_clahe)
    cv2.waitKey(2000)
    cv2.destroyAllWindows() 
    """
    """
    cv2.imshow("img_Hematoxilina",img_hematoxilina)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    """
    """
    cv2.imshow("img_gray",img_gray)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    """
    """
    cv2.imshow("img_filtrada",img_filtrada)
    cv2.waitKey(60000)
    cv2.destroyAllWindows() 
    """
    """
    cv2.imshow("img_thresholded",img_thresholded)
    cv2.waitKey(2000)
    cv2.destroyAllWindows() 
    """
    
    """
    cv2.imshow("img_watershed",img_rgb_w)
    cv2.waitKey(60000)
    cv2.destroyAllWindows() 
    """
    
    """
    cv2.imshow("img_blurred",img_blurred)
    cv2.waitKey(2000)
    cv2.destroyAllWindows() 
    """
    """
    cv2.imshow("img_matched",img_matched)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()  
    """

     
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
    
    
  
    