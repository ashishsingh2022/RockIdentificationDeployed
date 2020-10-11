import keras
keras.__version__

import tensorflow
tensorflow.__version__

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.8,zoom_range=0.8,horizontal_flip=True) #sher= distance increase flip,etc if not able to understand
test_datagen=ImageDataGenerator(rescale=1./255) # upper options can be used but not required

x_train=train_datagen.flow_from_directory(r"C:\Users\prana\Desktop\Project S\traindata",target_size=(64,64),batch_size=32,class_mode="categorical")
x_test=test_datagen.flow_from_directory(r"C:\Users\prana\Desktop\Project S\testdata",target_size=(64,64),batch_size=32,class_mode="categorical")


print(x_train.class_indices) #labels given to datasets

model=Sequential()

model.add(Convolution2D(128,(3,3),input_shape=(64,64,3),activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) #Input Layer Of ANN
model.add(Dense(units=64,init="random_uniform",activation="relu")) # Hidden Layer
model.add(Dense(units=64,init="random_uniform",activation="relu")) # Hidden Layer
model.add(Dense(units=64,init="random_uniform",activation="relu")) # Hidden Layer
model.add(Dense(units=13,init="random_uniform",activation="softmax")) # Output Layer
model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])
model.fit_generator(x_train,steps_per_epoch=709/32,epochs=5000,validation_data=x_test,validation_steps=237/32)

model.save("RockIdentification.h5")