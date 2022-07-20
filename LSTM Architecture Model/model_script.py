import shutil
import os
import cv2
import math
import numpy as np
import tensorflow as tf
import pandas as pd 
import skvideo.io
import mediapipe as mp
import glob
import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense


from helper_functions import Mediapipe_converter_testing,video_array_maker_testing

actions=np.array(["Hello","How are you","thank you"])


model = Sequential()

model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(45,258)))
model.add(LSTM(128,return_sequences=True, activation = 'relu'))
model.add(LSTM(256,return_sequences=True,activation="relu"))
model.add(LSTM(64, return_sequences = False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

print(model.summary())
model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

model.load_weights("model/170-0.83.hdf5")


for dirpath,dirnamed,files in os.walk("input_video"):
    for i,j,k in os.walk(dirpath):
      for index,m in enumerate(k):
        pather=os.path.join(i,m)
        print(f"Input Video Path : {pather}")
        video_array_maker_testing(pather,index)

for dirpath,dirnamed,files in os.walk("resized_videos"):
    for i,j,k in os.walk(dirpath):
      for index,m in enumerate(k):
        pather=os.path.join(i,m)
        print(f"Resized Video Path : {pather}")
        arr = Mediapipe_converter_testing(pather,index)
        arr= np.expand_dims(arr,axis=0)

        print(f" Video Array Shape : {arr.shape}")

        pred_prob=model.predict(arr)

        pred=np.argmax(pred_prob)

        print(f"predicted output : {actions[pred]}")
