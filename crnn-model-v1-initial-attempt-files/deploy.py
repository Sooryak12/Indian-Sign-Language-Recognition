import uvicorn
from fastapi import FastAPI,File,UploadFile
import skvideo.io
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM,Flatten, TimeDistributed, Conv2D, Dropout,Input
from notebooks.pipeline import video_array_maker
import os
import numpy as np

app=FastAPI()

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

classes=['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']


@app.route("/predict/video")
async def model_output(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("mp4", "MP4")
    if not extension:
        return "Video must be in Mp4 Format"
 
    video_array_maker(BytesIO(model_input),output_directory="",output_folder="No",remove_input=False)
    model_input=skvideo.io.vread(file.filename)
    prediction=model.predict(np.expand_dims(model_input,axis=0))    
    arg_pred=np.argmax(prediction,axis=1)
    
    return classes[arg_pred[0]]

if __name__ == "__main__":
    uvicorn.run(app, debug=True)