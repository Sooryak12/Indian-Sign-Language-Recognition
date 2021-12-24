import os
import skvideo.io
import cv2 
import time
import mediapipe as mp
import holistic
from holistic import pose_estimation

def video_array_maker(pather,height=224,width=224,output_directory="./Output",output_folder=pather.split("/")[3],remove_input=True):
    """
    pather : location of video
    height,width (default : 224) 
    output_directory :output main directory  (default : ./Output)
    output folder :output subdirectory
    """
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
#                arr.append(output)
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