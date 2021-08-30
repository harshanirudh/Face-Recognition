# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:06:04 2019

@author: Asus
"""
import os
import numpy as np
from PIL import Image
from numpy import savez_compressed
def extract_face(filename, required_size=(160, 160)):
    image=Image.open(filename)
    image=image.convert('RGB')
    face=np.asarray(image)
    return face

def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in os.listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        print(path)
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)
trainX, trainy = load_dataset('C:/Users/Asus/Desktop/project/Face_recog/Image_dataset/')
print(trainX.shape, trainy.shape)
savez_compressed('family-faces.npz', trainX, trainy)