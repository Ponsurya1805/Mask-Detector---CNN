##Import
import matplotlib.pyplot as plt
import os
import cv2
import PIL
from PIL import Image, ImageChops
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
from tensorflow import keras
from keras import layers,models
from keras.models import Sequential
from google.colab import drive

###Loading and Classifying Training Data
data_dir=pathlib.Path("/TrainData/")
len(list(data_dir.glob('*/*.png')))
Lable={
    "Mask":list(data_dir.glob('Mask/*.png')),
    "No Mask": list(data_dir.glob('No Mask/*.png'))
}


###Labelling Data
entry = {
    "Mask":0,
    "No Mask":1
}


###Data Processing
X,Y=[],[]
for Lab,images in Lable.items():
  for image in images:
    #print(image)
    im = Image.open(image)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
      imd=im.crop(bbox)
      imd=cv2.cvtColor(np.float32(imd), cv2.COLOR_RGB2BGR)
    re_imag= cv2.resize(imd,(224,224))
    X.append(re_imag)
    Y.append(entry[Lab])
    print(len(Y))

###Train,Test, Validation    
X=np.array(X)
Y=np.array(Y)
Xtrain,Xval,Ytrain,Yval=train_test_split(X,Y,test_size=0.1,random_state=10)
Xtra,Xtes,Ytra,Ytes=train_test_split(Xtrain,Ytrain,test_size=0.2,random_state=10)

###Scaling X data 
Xtra_scaled=Xtra/255
Xtes_scaled=Xtes/255


###Buliding  Model

model = Sequential()
##CNN
#model.add(dataaug)
model.add(layers.Conv2D(input_shape=(224,224,3),filters=10,kernel_size=(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=12,kernel_size=(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=20,kernel_size=(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))

##Dense
model.add(layers.Flatten())
model.add(layers.Dense(5,activation="relu"))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(3,activation="relu"))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(1,activation="sigmoid"))

opt=keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])
H=model.fit(Xtra_scaled,Ytra,epochs=10)


##Plotting Output
N = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
