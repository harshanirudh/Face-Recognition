# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:17:07 2019

@author: Asus
"""

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

model=load_model('facenet_keras.h5')
model1=joblib.load('svm-face-classifier.sav')
data=np.load('family-faces-embeddings.npz')
y=data['arr_1']
out_encoder = LabelEncoder()
out_encoder.fit(y)
y=out_encoder.transform(y)
detector = MTCNN()
#while True: 
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    #Capture frame-by-frame
    __, frame = cap.read()
#    cv2.namedWindow("frame",cv2.WND_PROP_FULLSCREEN)
#    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    fram_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #Use MTCNN to detect faces
    pixels=np.asarray(fram_rgb)
    result = detector.detect_faces(fram_rgb)
    #for results in result:
        #print ()
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            x1, y1, width, height = bounding_box
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize((160,160))
            face_array = np.asarray(image)
            face_embedding=get_embedding(model,face_array)
            in_encoder = Normalizer(norm='l2')
            face_embedding=in_encoder.transform(face_embedding.reshape(1,-1))
            name=model1.predict(face_embedding)
            prob_array=model1.predict_proba(face_embedding)[0]
            prob=prob_array[name]
            if prob>0.5:
                name=out_encoder.inverse_transform(name)
                name=name[0]
            else:
                name="unknown"
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          1)
    
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
            cv2.putText(frame,str(name),(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(200,255,155),2,cv2.LINE_AA)
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()

 