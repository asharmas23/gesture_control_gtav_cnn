import numpy as np
import cv2
import os

path='C:/Users/Aman Sharma/Desktop/GESTURE CONTROL/DATASET/One Train'


con = 0
delay = 0
i = 0
cap = cv2.VideoCapture(0)
cap.set(3,376)
cap.set(4,240)
ret,frame = cap.read()
width = cap.get(3)
height = cap.get(4)
#fps = cap.get(5)
x = int(width*0.30)
y = int(height*0.02)
w = int(width *0.70)
h = int(height *0.45) 

#kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel_sharpening = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

kernel = np.ones((5,5),np.uint8)

crop_img = frame[y:h,x:w]  

while True:
    _,frame = cap.read()
    cv2.rectangle(frame,(x,y),(w,h),(0,255,0),5)
    cv2.imshow('Op',frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.dilate(gray,None)
    crop_img = gray[y:h,x:w]  
    average = np.float32(crop_img)
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        print('Capturing Frames Starting Soon...')
        con = 1
    
    
    if con == 1:
        crop_img = gray[y:h,x:w]
        a,b = crop_img.shape
        if delay<150:
            cv2.accumulateWeighted(crop_img,average,0.01)
            background = cv2.convertScaleAbs(average)
            #cv2.imshow('Input',frame)
            cv2.imshow('Disappearing Background',background)
#            cv2.imshow('OUTPUT',crop_img)
        delay = delay + 1
        if 150 <= delay <= 200:
            print('Get ready! Starting in {}',200-delay)
            
        if delay > 200:
            print('Capturing')
            img_name = str(i)+'.jpg'
            mask = cv2.absdiff(background,crop_img)
            mask = cv2.convertScaleAbs(mask)
            #mask = cv2.GaussianBlur(mask,(7,7),0)
            mask = cv2.filter2D(mask,-1,kernel_sharpening)
            mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#            contours = cv2.findContours(mask[1],cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#            contours = max(contours[1],key = cv2.contourArea)
#            for c in contours[1]:
#                accuracy = 0.01 * cv2.arcLength(c,True)
#                approx = cv2.approxPolyDP(c,accuracy,True)
#                cv2.drawContours(frame,[approx + (x,y)],-1,(0,0,255),1)
##                cv2.drawContours(mask[1],[approx],-1,(0,0,255),2)
#                cv2.imshow('MASK',frame)
            cv2.imwrite(os.path.join(path,img_name),mask[1])
            i = i + 1
            if i == 100:
                print('frame capturing Completed')
                con = 0
                break
        
            
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()