import pandas as pd
import numpy as np
import cv2 
import skvideo.io
import os
import tensorflow as tf
from notebooks.pipeline import video_array_maker
import time


#Loading model (path will change)
model=tf.keras.models.load_model(r"models\full_model")

classes=['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

try:
    os.mkdir("out")
except:
    pass

num_of_videos=5
i=0
while(i<num_of_videos):
    cap = cv2.VideoCapture(0)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out= cv2.VideoWriter('input.mp4', cv2.VideoWriter_fourcc(*'DIVX'),10, (width,height))
    start= time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
           break 
        out.write(frame)
        if time.time() -start >3:
            break     

    cap.release()        
    out.release()      
    cv2.destroyAllWindows()

    outpath=video_array_maker("input.mp4",output_directory="out",output_folder="No",remove_input=False) 

    model_input=skvideo.io.vread(outpath)
    prediction=model.predict(np.expand_dims(model_input,axis=0))
    arg_pred=np.argmax(prediction,axis=1)
    i+=1
    print("{} video is {}".format(i,classes[arg_pred[0]]))
    