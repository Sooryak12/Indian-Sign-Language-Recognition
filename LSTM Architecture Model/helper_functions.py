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
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

  
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

def pose_estimation(image,results):
        
        # 1. Draw face landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
        #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                         )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(102,255,51), thickness=3, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(102,255,51), thickness=3, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                                 )
                        
        return image

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose,lh,rh])

def video_array_maker_testing(pather, video_num,height=224,width=224,output_directory="resized_videos",output_folder="No",remove_input=False):
    """
        pather : location of video
        height,width (default : 224) 
        output_directory :output main directory  (default : ./Output)
        output folder :output subdirectory
    """

    if output_folder == "No":
            output_folder=""  
    videodata = skvideo.io.vread(pather)  
    outpath=os.path.join(output_directory,output_folder,os.path.split(pather)[1])
    out = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('M','J','P','G'), 10,(videodata.shape[2],videodata.shape[1]))

    actualframe=len(videodata)
     
    if actualframe >=45:
          for i in range (actualframe):
            x=round (actualframe/(45)  * i)
            if x >=actualframe:
                    break
            else:
                frame =videodata[x]             
                #output=cv2.resize(frame,(videodata.shape[2],videodata.shape[1]),interpolation=cv2.INTER_NEAREST)
                output =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                out.write(output)
                
                         
    else:
          for i in range(actualframe):
              frame=videodata[i]
              #output=cv2.resize(frame,(videodata.shape[2],videodata.shape[1]),interpolation=cv2.INTER_NEAREST)
              output =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
              out.write(output)
    
          for i in range(45-actualframe):
              newframe=np.zeros(shape=(videodata.shape[2],videodata.shape[1],3))
              #frame=videodata[i] 
              out.write(np.uint8(newframe))
           
    out.release()
   
    if remove_input==True:
        os.remove(pather)
        
        
def Mediapipe_converter_testing(pather,video_num,output_directory="mediapipe_video",output_folder="No",remove_input=False):
  """
    pather : location of video
    height,width (default : 224) 
    output_directory :output main directory  (default : ./Output)
    output folder :output subdirectory
  """


  if output_folder == "No":
    output_folder=""  
  videodata = skvideo.io.vread(pather)  


  outpath=os.path.join(output_directory,output_folder,os.path.split(pather)[1])
  print(f"mediapipe inscribed video path location  : {outpath}")
  out = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('M','J','P','G'), 10,(224,224))
  np_array=[]
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for x in range(45):
                 frame=videodata[x]
                 image,results = mediapipe_detection(frame,holistic)
                 output=pose_estimation(frame,results)
                 keypoints = extract_keypoints(results)
                 np_array.append(keypoints)

                 output=cv2.resize(output,(224,224),interpolation=cv2.INTER_AREA)
                 output =cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
                 out.write(output)
    np.save(f"np_array/{video_num}",np_array)

    out.release()
   
    if remove_input==True:
        os.remove(pather)

    np_array=np.array(np_array)

    return np_array