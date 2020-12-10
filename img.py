#tensorflow 2.2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

#load models
facenet=cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
model=load_model('face_detector/mask_detector.model')

#load image
img=cv2.imread('imgs/00t.jpg')#이미지 저장
h,w=img.shape[:2]#이미지의 높이,너비 저장

plt.figure(figsize=(16,10))
plt.imshow(img[:,:,::-1])#이미지가 잘 저장됐는지 확인(opencv로 영상을 읽으면 BGR로 읽힘 이것을 RGB로 바꿔줌)


#Preporcess Image for Face Detection

#dnn 모듈이 사용하는 형태로 이미지를 변형
blob=cv2.dnn.blobFromImage(img,scalefactor=1.,size=(300,300),mean=(104.,177.,123.))
facenet.setInput(blob)#변형된 이미지를 facenet의 input으로 설정
dets=facenet.forward()#forward()를 사용하여 결과 추론하고 dets에 저장

#Detect Faces
faces=[]

for i in range(dets.shape[2]):
    confidence=dets[0,0,i,2]
    if confidence<0.5: #threshold=0.5, 0.5미만인 경우는 모두 넘김
        continue
#x와 y의 boundingbox를 구해준다.
    x1=int(dets[0,0,i,3]*w)
    y1=int(dets[0,0,i,4]*h)
    x2=int(dets[0,0,i,5]*w)
    y2=int(dets[0,0,i,6]*h)

    #bounding box를 가지고 원본 이미지에서 얼굴만 잘라낸다.
    face=img[y1:y2,x1:x2]
    faces.append(face)#faces에 얼굴만 자른 이미지를 저장

plt.figure(figsize=(16,5))

#faces에 저장된 얼굴을 다시 한번 찍어준다
for i,face in enumerate(faces):
    plt.subplot(1,len(faces),i+1)
    plt.imshow(face[:,:,::-1])

#Detect Masks from Faces
plt.figure(figsize=(16,5))

#face의 갯수만큼 loop를 돌면서 마스크 착용 여부를 확인
for i, face in enumerate(faces):
    face_input = cv2.resize(face, dsize=(224, 224))
    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)#이미지의 컬러시스템을 변경(BGR->RGB)
    face_input = preprocess_input(face_input)
    face_input = np.expand_dims(face_input, axis=0)
    #0번 axis에 차원을 하나 더 추가해준다.((224,224,3)->(0,224,224,3))

    mask, nomask = model.predict(face_input).squeeze()

    plt.subplot(1, len(faces), i + 1)
    plt.imshow(face[:, :, ::-1])
    plt.title('%.2f%%' % (mask * 100))