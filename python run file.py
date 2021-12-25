input_file_path =""   #specify the video's file path here

import pandas as pd
import numpy as np
import mediapipe as mp
import skvideo.io
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM,Flatten, TimeDistributed, Conv2D, Dropout,Input
from notebooks.pipeline import video_array_maker

#Loading Model
mobilenet = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224,224,3),
        weights='imagenet')

model=Sequential()
model.add(TimeDistributed(mobilenet ,input_shape=(45,224,224,3)))
model.add(TimeDistributed( Flatten()))
model.add(LSTM(128, activation='relu', return_sequences=False)) 
model.add(Dense(64,activation="relu"))
model.add(Dense(16,activation="softmax")) 
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=tf.keras.metrics.Accuracy())

model.load_weights("checkpoint/checkpoint")

print("model loaded")

classes=['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']
try:
    os.mkdir("out")
except:
    pass

#Pipeline:
video_array_maker(input_video_path,output_directory="out",output_folder="No",remove_input=False) 
# Creates a pose points embedded video in 45*224*224*3 shape (45 frames)
print("Points Embedded")
#Reading Output Video
model_input=skvideo.io.vread(os.path.join("out",input_video_path))

#Predicting the Action
prediction=model.predict(np.expand_dims(model_input,axis=0))

#Printing Class
arg_pred=np.argmax(prediction,axis=1)
print(arg_pred)
print(classes[arg_pred[0]])



