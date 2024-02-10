import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from directkeys import PressKey,ReleaseKey, W, A, S, D,E,F

path='C:/Users/Aman Sharma/Desktop/GESTURE CONTROL/DATASET/TEST/'


test_datagen = ImageDataGenerator(rescale = 1./255)

model = load_model('GestureModel.h5')

res = []
cap = cv2.VideoCapture(0)
cap.set(3,376)
cap.set(4,240)
ret,frame = cap.read()
width = cap.get(3)
height = cap.get(4)
fps = cap.get(5)
x = int(width*0.30)
y = int(height*0.02)
w = int(width *0.70)
h = int(height *0.45)
con = 0
delay = 0 
i = 0

kernel_sharpening = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

kernel = np.ones((5,5),np.uint8)

crop_img = frame[y:h,x:w]  


while True:
    _,frame = cap.read()
    cv2.rectangle(frame,(x,y),(w,h),(0,255,0),5)
    cv2.imshow('Captured Frame',frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    crop_img = frame[y:h,x:w]  
    average = np.float32(crop_img)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        print('Capturing Frames Starting Soon...')
        con = 1
    
    
    if con == 1:
        crop_img = frame[y:h,x:w]
        a,b = crop_img.shape
        delay = delay + 1

        if delay<150:
            cv2.accumulateWeighted(crop_img,average,0.01)
            background = cv2.convertScaleAbs(average)
            #cv2.imshow('Input',frame)
            cv2.imshow('Disappearing Background',background)
            cv2.imshow('Crop Image',crop_img)
        if 150 <= delay <= 200:
            print('Get ready! Starting in {}',200-delay)
            
        if delay > 200:
            print('DETECTING!')
            img_name = str(i)+'.jpg'
            mask = cv2.absdiff(background,crop_img)
            #mask = cv2.GaussianBlur(mask,(7,7),0)
            mask = cv2.filter2D(mask,-1,kernel_sharpening)
            mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #cv2.imshow('MASK',mask)
            cv2.imwrite(os.path.join(path,img_name),mask[1])
            if i == 50:
                i = 0
            #cv2.imshow('MASK',mask)
            test_img = image.load_img(path + str(i) + '.jpg',target_size = (128,104),color_mode = 'grayscale')
            test_img = image.img_to_array(test_img)
            test_img = np.expand_dims(test_img,axis = 0)
            res = model.predict(test_img)
            res = res.astype(int)
            if (res == ([[1, 0, 0, 0, 0, 0, 0, 0]])).all():
                className = 'Fist (EXIT)'
                PressKey(F)
            elif (res == ([[0, 1, 0, 0, 0, 0, 0, 0]])).all():
                className = 'One (FORWARD)'
                PressKey(W)
            elif (res == ([[0, 0, 1, 0, 0, 0, 0, 0]])).all():
                className = 'Noise (NO INPUT)'
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                ReleaseKey(E)
                ReleaseKey(F)
            elif (res == ([[0, 0, 0, 1, 0, 0, 0, 0]])).all():
                className = 'Palm (REVERSE)'
                PressKey(S)
            elif (res == ([[0, 0, 0, 0, 1, 0, 0, 0]])).all():
                className = 'Swing (LEFT)'
                PressKey(W)
                PressKey(A)
            elif (res == ([[0, 0, 0, 0, 0, 1, 0, 0]])).all():
                className = 'Three (EXIT)'
                PressKey(F)
            elif (res == ([[0, 0, 0, 0, 0, 0, 1, 0]])).all():
                className = 'Two (RIGHT)'
                PressKey(W)
                PressKey(D)
            elif (res == ([[0, 0, 0, 0, 0, 0, 0, 1]])).all():
                className = 'Yo! (HORN)'
                PressKey(E)
            print(className)
            i = i + 1
            if i % 10 == 0:
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                ReleaseKey(E)
                ReleaseKey(F)
                
#i = 0
#test_img = image.load_img(path + str(i) + '.jpg',target_size = (128,104),color_mode = 'grayscale')
#test_img = image.img_to_array(test_img)
#test_img = np.expand_dims(test_img,axis = 0)
#res = model.predict(test_img)            
#frame = cv2.imread('0.jpg')
#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#frame = standardize(frame)
#res = mqodel.predict(frame)     
#

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()