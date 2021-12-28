from tensorflow import keras
from keras import models
import cv2
import numpy as np
import cv2

entry = {
    "0":"Mask",
    "1":"No Mask"
}
facenet=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mod=models.load_model(r"Maskdetect1.model")
color_dict={0:(0,255,0),1:(0,0,255)}


Cap=cv2.VideoCapture(0)
width = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
  ret,image=Cap.read()
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  #img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  Faces=facenet.detectMultiScale(gray,1.3,5)
  for x,y,w,h in Faces:
    Face_img=gray[y:y+w,x:x+w]
    resize=cv2.resize(image, (224,224))
    print(resize.shape)
    norm=resize/255
    #print(norm)
    norm_reshaped=np.reshape(norm,(1,224,224,3))
    result=mod.predict(norm_reshaped)
    print(result)
    if result < 0.5:
      t=0
      f="Mask"
      #yt.append(t)
    elif result > 0.5 :
      t=1
      f="No Mask"
    lable=np.argmax(result)
    #print(lable)
    cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[t],2)
    cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[t],-1)
    cv2.putText(image,f,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
  cv2.imshow("Live",image)
  k = cv2.waitKey(1) & 0xFF
  if k == ord('q'):
    break
Cap.release()
cv2.destroyAllWindows()
print("Bye")
