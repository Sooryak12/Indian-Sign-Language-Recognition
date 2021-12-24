import os
import skvideo.io
import cv2 
import time
import mediapipe as mp

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




def video_array_maker(pather,height=224,width=224,output_directory="./Output",output_folder=None,remove_input=True):
  """
    pather : location of video
    height,width (default : 224) 
    output_directory :output main directory  (default : ./Output)
    output folder :output subdirectory
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
  #    arr=[]
    actualframe=len(videodata)
    #print("Actual frame {}".format(actualframe))
    if actualframe >=45:
          for i in range (actualframe):
            x=round (actualframe/(45)  * i)
            if x >=actualframe:
                    break
            else:
                frame =videodata[x]
                #frame =cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                results = holistic.process(frame)  

                output = pose_estimation(frame,results)
                output =cv2.resize(output,(width,height),interpolation=cv2.INTER_AREA)
                output =cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
                #arr.append(output)
                out.write(output)               
    else:
          for i in range(actualframe):
              frame=videodata[i]
              frame=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
              results = holistic.process(frame)  
              output = pose_estimation(frame,results)
              output =cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
              out.write(output)
          for i in range(45-actualframe):
              
              newframe=np.zeros(shape=(height,width,3))
              

              out.write(np.uint8(newframe))
    out.release()
    #print("File Created : {}".format(outpath))
    if remove_input==True:
        os.remove(pather)