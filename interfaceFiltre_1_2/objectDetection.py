##optimisé
#vert
#boule
#taille de frame en entrée

import cv2
import numpy as np
import time

lo = np.array([60,50,50])
hi = np.array([180,255,255])
def erode(mask,kernel):
    ym,xm=kernel.shape
    yi,xi=mask.shape
    m=xm//2
    mask2=mask.copy()
    for y in range(yi):
        for x in range(xi):
            if mask[y,x]==255:
                if  ( y<m or y>(yi-1-m) or x<m or x>(xi-1-m)):
                    mask2[y,x]=0
                else:
                    v=mask[y-m:y+m+1,x-m:x+m+1] 
                    for h in range(0,ym):
                        for w in range(0,xm): 
                            if(v[h,w]<kernel[h,w]):
                                mask2[y,x]=0
                                break
                        if(mask2[y,x]==0): 
                            break
    return mask2

# def erode(img,kernel):
    ym,xm=kernel.shape
    xi,yi=img.shape
    m=xm//2
    mask2=np.zeros(img.shape,img.dtype)
    for y in range(xi):
        for x in range(yi):   
             if not(y<m or y>(xi-1-m) or x<m or x>(yi-1-m)):
                v=img[y-m:y+m+1,x-m:x+m+1] 
                b=False
                for h in range(ym):
                     for w in range(xm): 
                        if(v[h,w]<kernel[h,w]):
                             b=True
                             break
                     if(b): 
                         break
                if(not b): 
                    mask2[y,x]=255
    return mask2
def inRange(img,lo,hi):
    mask=np.zeros((img.shape[0],img.shape[1]))
    for y in range(img.shape[0]):
         for x in range(img.shape[1]):
              if(img[y,x,0]<=hi[0] and img[y,x,0]>=lo[0] and img[y,x,2]<=hi[2] and img[y,x,2]>=lo[2] and img[y,x,1]<=hi[1] and img[y,x,1]>=lo[1] ):
                    mask[y,x]=255
    return mask
def center(img):
    b=True
    c=True
    premiery=0
    derniery=img.shape[0]
    premierx=0
    dernierx=img.shape[1]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(b and img[y,x]==255):
                b=False
                premiery=y
            if( c and img[img.shape[0]-y-1,x]==255):
                c=False
                derniery=img.shape[0]-y-1
            if(not b and not c):
                break
        if(not b and not c):
            break    
    b=True
    c=True
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if(img[y,x]==255 and b):
                b=False
                premierx=x
            if(img[y,img.shape[1]-x-1]==255 and c):
                c=False
                dernierx=img.shape[1]-x-1
            if(not b and not c):
                break
        if(not b and not c):
            break

    x=((dernierx-premierx)/2)+ premierx
    y=((derniery-premiery)/2)+ premiery
    return (int(x),int(y))
def detect_inrange(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = inRange(image,lo,hi)
    mask = erode(mask,kernel=np.ones((5,5)))
    return mask
def resize(img):
     img2=np.zeros(((int(img.shape[0]/2.5))+1,(int(img.shape[1]//2.5))+1,3),img.dtype)
     for y in range(0,int(img.shape[0]/2.5)):
         for x in range(0,int(img.shape[1]/2.5)):
             img2[y,x,:]=img[int(y*2.5),int(x*2.5),:]
   
     return img2
def object_Detection_Color():   
    VideoCap = cv2.VideoCapture(0)
    while True:
        ret, frame = VideoCap.read()
        frame=resize(frame)
        cv2.flip(frame,1, frame)
        mask = detect_inrange(frame)
        centre=center(mask)
        if mask is not None:
            cv2.circle(frame, centre, 5, (0, 0, 255),-1)
            #cv2.imshow('mask', mask)
            #cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    VideoCap.release()
    cv2.destroyAllWindows()
#object_Detection_Color()





