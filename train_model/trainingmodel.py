import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='elu', input_shape=(128, 104, 1),padding = 'same'))
model.add(Conv2D(64, (5, 5), activation='elu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (7, 7), activation='elu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), activation='elu',padding = 'same'))
model.add(Conv2D(32, (3, 3), activation='elu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(256, (3, 3), activation='elu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(128, (3, 3), activation='elu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3), activation='elu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (3, 3), activation='elu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())


model.add(Dense(1500,activation='elu'))

model.add(Dense(8,activation='softmax'))

optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics = ['accuracy'])
#
#loadedImages = []
#testImages = []
#outputVectors = []
#testLabels = []

path='C:/Users/Aman Sharma/Desktop/GESTURE CONTROL/DATASET'
#path='DATASET/DATA'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(path + '/Training_Set',
                                                 target_size = (128, 104),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',            
                                                 color_mode = 'grayscale')

test_set = test_datagen.flow_from_directory(path + '/Test_Set',
                                            target_size = (128, 104),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            color_mode = 'grayscale')

#for i in range(0,799):
#                                        outputVectors.append([1,0,0,0,0,0,0,0])
#for i in range(0,799):
#                                        outputVectors.append([0,1,0,0,0,0,0,0])
#for i in range(0,799):
#                                        outputVectors.append([0,0,1,0,0,0,0,0])
#for i in range(0,799):
#                                        outputVectors.append([0,0,0,1,0,0,0,0])
#for i in range(0,799):
#                                        outputVectors.append([0,0,0,0,1,0,0,0])
#for i in range(0,799):
#                                        outputVectors.append([0,0,0,0,0,1,0,0])
#for i in range(0,799):
#                                        outputVectors.append([0,0,0,0,0,0,1,0])
#for i in range(0,799):
#                                        outputVectors.append([0,0,0,0,0,0,0,1])
#                
#
#
#for i in range(0,199):
#                                        testLabels.append([1,0,0,0,0,0,0,0])
#for i in range(0,199):
#                                        testLabels.append([0,1,0,0,0,0,0,0])
#for i in range(0,199):
#                                        testLabels.append([0,0,1,0,0,0,0,0])
#for i in range(0,199):
#                                        testLabels.append([0,0,0,1,0,0,0,0])
#for i in range(0,199):
#                                        testLabels.append([0,0,0,0,1,0,0,0])
#for i in range(0,199):
#                                        testLabels.append([0,0,0,0,0,1,0,0])
#for i in range(0,199):
#                                        testLabels.append([0,0,0,0,0,0,1,0])
#for i in range(0,199):
#                                        testLabels.append([0,0,0,0,0,0,0,1])  
#                                        
                                
test_set.class_indices

history = model.fit_generator(training_set, steps_per_epoch = 6400,
                         epochs = 25,verbose = 1,shuffle = True,
                         use_multiprocessing = True,
                         validation_data = test_set,
                         validation_steps = 1600)
        
#     validation_split = 0.2


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
model.save('MyModel.h5')
print('Model saved!')




