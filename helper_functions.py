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




def mediapipe_detection(image,model):
    """
        function which taks in image and feeds it to the mediapipe posenet detection model to extract key body points.
            inputs 
                image : image to process
                model  : model to be used to process the image """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

mp_holistic=mp.solutions.holistic

def extract_keypoints(results):
    """ function which takes in coordinated of body ,left hand ,right hand points and creates a concatanated numpy array 
        inputs
            results : coordinate information from mediapipe model"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh,rh])

def convert_video_to_pose_embedded_np_array(pather,remove_input=False):
    """
        function which takes in video , evenly chooses 45 frames and feeds these frames to mediapipe model and extract 
        key positions as numpy array.


        Function workflow :
            1. takes in video as input.
            2. calculates no. of frames in the video.
                2.1 if frames >45 algorithm chooses the evenly spaced frames to avoid loss in information.
                2.2 if frames <45 adds empty numpy array to keep the input size same.

        Input
            pather : location of video
            remove_input : True - delete the created video . False - to keep it.

        returns numpy array with key coordinates of each frame
    """

    videodata = skvideo.io.vread(pather)  
    np_array=[]
    actualframe=len(videodata)
    print(actualframe)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
     
     if actualframe >=45:
            
              for i in range (actualframe):
                x=round (actualframe/(45)  * i)  #evenly choosing frames
                if x >=actualframe:
                        break
                else:
                    frame =videodata[x]             
                    image,results = mediapipe_detection(frame,holistic) #applying mediapipe model
                    keypoints = extract_keypoints(results) #concatanates body , left hand , right hand points.
                    np_array.append(keypoints) # save as a single numpy array
                    
                    
     else:
              key_points_shape = 0
              for i in range(actualframe):
                  frame=videodata[i]
                  image,results = mediapipe_detection(frame,holistic) #applying mediapipe model
                  keypoints = extract_keypoints(results) #concatanates body , left hand , right hand points.
                  key_points_shape = keypoints.shape()  # save as a single numpy array
                  np_array.append(keypoints)
        
              for i in range(45-actualframe):
                  np_array.append(np.zeros(shape=key_points_shape)) # add empty frames at end to keep array size same.
    
     np.save("np_array_0",np_array)

     if remove_input==True:
          os.remove(pather)
    
     np_array=np.array(np_array)

     return np_array
        
        
