# Command Line :  python lstm_model_run_through_cmd_line.py -i input_file_path

import os
import cv2
import time
import numpy as np

import argparse
from argparse import ArgumentParser

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

from helper_functions import convert_video_to_pose_embedded_np_array


def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

if __name__ == '__main__':
    parser = ArgumentParser(description="input video from Command line.")
    parser.add_argument("-i", "--input", dest="filename", required=True, type=validate_file,
                        help="input file", metavar="FILE")
    args = parser.parse_args()
    
    input_video_path =args.filename
    print(args.filename)

    
    #Loading Model
    def initialize_model():
        """ Initializes lstm model and loads the trained model weight  """
        model = Sequential()
        model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(45,258)))
        model.add(LSTM(128,return_sequences=True, activation = 'relu'))
        model.add(LSTM(256,return_sequences=True,activation="relu"))
        model.add(LSTM(64, return_sequences = False,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation = 'relu'))
        model.add(Dense(3,activation='softmax'))

        print(model.summary())
        model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

        model.load_weights(r"lstm-model\170-0.83.hdf5")

        return model

    model = initialize_model()

    #output words to be predicted
    actions=np.array(["Hello","How are you","thank you"])
    
    out_np_array=convert_video_to_pose_embedded_np_array(input_video_path,remove_input=False) #function to detect key points in each frame and return them as an numpy array.
    
    prediction=model.predict(np.expand_dims(out_np_array,axis=0))  # predict with the model
    arg_pred=np.argmax(prediction,axis=1)

    print("video is {}".format(actions[arg_pred[0]])) # print the predicted class

