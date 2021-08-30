# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:49:43 2019

@author: Asus
"""
import numpy as np
from keras.models import load_model
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
data=np.load('family-faces.npz')
trainX,trainy=data['arr_0'],data['arr_1']
print("loaded trainx,trainy",trainX.shape,trainy.shape)
model=load_model('facenet_keras.h5')
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
np.savez_compressed('family-faces-embeddings.npz', newTrainX, trainy)