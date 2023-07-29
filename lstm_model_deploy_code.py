import os
import cv2
import time
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

from helper_functions import convert_video_to_pose_embedded_np_array

#output words to be predicted
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

print("model loaded")

try:
    # Folder to save the input video saved during video capture.
    os.mkdir("input-video")  
except:
    pass

num_of_videos=10  #parameter (can be modified)
i=0

while(i<num_of_videos):
    cap = cv2.VideoCapture(0) # reads the camera
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video_start")
    out= cv2.VideoWriter('input-video\input.mp4', cv2.VideoWriter_fourcc(*'DIVX'),10, (width,height)) # creates video file to save the captured video
    
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read() # read frame from camera
        if not ret:
           break 
        out.write(frame) # saves frame in file
        if time.time() - start >5:  # 5 second video to capture the action
            break     

    cap.release()        
    out.release()      
    cv2.destroyAllWindows()
    
    print("video_made")

    out_np_array=convert_video_to_pose_embedded_np_array("input-video\input.mp4",remove_input=False) #function to detect key points in each frame and return them as an numpy array.
    
    prediction=model.predict(np.expand_dims(out_np_array,axis=0))
    arg_pred=np.argmax(prediction,axis=1)

    i+=1
    print("{} video is {}".format(i,actions[arg_pred[0]]))
