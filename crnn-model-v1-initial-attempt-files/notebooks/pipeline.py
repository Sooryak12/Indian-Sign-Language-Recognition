import os
import skvideo.io
import cv2 
import mediapipe as mp
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

def pose_estimation(image,results):
        """ Function which takes in image , and the result from mediapipe posenet and uses those cooridnates 
        to mark coordinates in yellow color in frames"""
        
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




def video_array_maker(pather,height=224,width=224,output_directory="./Output",output_folder=None,remove_input=True):
  """
    pather : location of video
    height,width (default : 224) 
    output_directory :output main directory  (default : ./Output)
    output folder :output subdirectory

    the objective of this function is to take in video , choose evenly spaced 45 frames to keep the input shape same and not loose out information
    apply posenet detection and embed coordinates in the frames and return an numpy array.

  """
  if output_folder is None:
    output_folder=pather.split("/")[3]
  elif output_folder == "No":
    output_folder=""  
  videodata = skvideo.io.vread(pather)  
  outpath=os.path.join(output_directory,output_folder,os.path.split(pather)[1])
  out = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
  #start=time.time()

  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    actualframe=len(videodata)
    #print("Actual frame {}".format(actualframe))
    if actualframe >=45:
          for i in range (actualframe):
            x=round (actualframe/(45)  * i)  # evenly chooses 45 frames from the frames in video
            if x >=actualframe:
                    break
            else:
                frame =videodata[x]

                results = holistic.process(frame)  

                output = pose_estimation(frame,results)   # returns frame with key points embedded  in it.
                output =cv2.resize(output,(width,height),interpolation=cv2.INTER_AREA)  #resize frame 
                output =cv2.cvtColor(output,cv2.COLOR_BGR2RGB)  #convert BGR to RGB
                #arr.append(output)
                out.write(output)               
    else:
          for i in range(actualframe):
              frame=videodata[i]

              results = holistic.process(frame)  
              output = pose_estimation(frame,results)    # returns frame with key points embedded  in it.
              output=cv2.resize(output,(width,height),interpolation=cv2.INTER_AREA)  #resize frame
              output =cv2.cvtColor(output,cv2.COLOR_BGR2RGB) #convert BGR to RGB
              out.write(output)
          for i in range(45-actualframe):
              
              newframe=np.zeros(shape=(height,width,3)) # if no. of frames <45 , add empty frames.
              

              out.write(np.uint8(newframe))
    out.release()
    #print("File Created : {}".format(outpath))
    if remove_input==True:
        os.remove(pather)