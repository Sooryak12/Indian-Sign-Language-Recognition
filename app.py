# app.py - FastAPI web application for action recognition from uploaded videos
# This script defines a FastAPI web application that allows users to upload videos
# and get action recognition predictions using a pre-trained LSTM model.

from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
import os
import skvideo.io
import uvicorn
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

from helper_functions import convert_video_to_pose_embedded_np_array

app = FastAPI()

actions=np.array(["Hello","How are you","thank you"])

def initialize_model():
    """ Initializes lstm model and loads the trained model weight  """
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

    model.load_weights(r"lstm-model\170-0.83.hdf5")

    return model

model = initialize_model()

@app.post("/upload-video/")
def upload_video(file: UploadFile = File(...)):
    """ adds video format to video recieved from server and predicts the action made in the video with the trained model 
        returns : the prediction action"""
    video_format = os.path.splitext(file.filename)[1]

    # Check if the video format is valid (you might want to check against a list of accepted formats)
    if video_format not in ['.mp4', '.avi', '.mov']:
        return {"error": "Invalid video format"}

    # Save the uploaded video with the correct extension
    temp = NamedTemporaryFile(suffix=video_format, delete=False)
    try:
        
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);

            print(temp.name)
            out_np_array=convert_video_to_pose_embedded_np_array(temp.name,remove_input=False) #function to detect key points in each frame and return them as an numpy array.
    
            prediction=model.predict(np.expand_dims(out_np_array,axis=0))
            arg_pred=np.argmax(prediction,axis=1)
            
        except Exception:
                    return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        #temp.close()  # the `with` statement above takes care of closing the file
        os.remove(temp.name)
        
    return actions[arg_pred[0]]

@app.get("/test/")
def test():
    return "working"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)