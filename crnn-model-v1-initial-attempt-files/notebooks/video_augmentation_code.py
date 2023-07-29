import cv2
import os
import imgaug as ia
import imageio
from imgaug import augmenters as iaa
import shutil
import argparse
import random
import time
from vidaug import augmentors as va
from multiprocessing import Pool
import concurrent.futures

main_folder_path=""
output_folder_path=""
no_of_clips_to_augment_per_frame=0
video_clip_names=[]

def augment_and_save_frames(video_reader,output_folder_path,video_clip_name,i,fps,w,h):
    """
        Fetch each frame of video and augment and save as picture in a temporary folder
        Args:
            video_reader: Video reader object
            rotation_angle: int (Angle of rotation of image)
            noise_value: int (noise value between 0 to 100)
            temp_folder_path: string (temporary path to store video frames)
            output_folder_path: string (output folder path)
            video_clip_name: string (video name)
            i: no of clip augmented
    """
    
    # These 4 lines take care of abnormal file names
    temp = video_clip_name.replace(" ","")
    temp = temp.split(".")
    editted_name = temp[0]+"_"+str(i)+"."+temp[1]
    path_of_video_to_save = output_folder_path+"//"+editted_name
    # Noise value to add to videos for augmentation
#   noise_value = random.randint(0,60)
    if i%2==0:
        flip=True
    else:
        flip=False
    noise_value = 0
    # Rotation angle for video augmentation
    rotation_angle = random.randint(-8,8)
    #print("Rotation angle for augmented clip is ", rotation_angle)
    #print("Noise value to add to augmented clip is ", noise_value)

    print(editted_name, rotation_angle, "degrees")
    seq = iaa.Sequential([
        iaa.Fliplr(flip),
        iaa.Affine(rotate=rotation_angle)
    ])

  

    fourcc = 'mp4v'  # output video codec
    video_writer = cv2.VideoWriter(path_of_video_to_save, cv2.VideoWriter_fourcc(*fourcc),fps,(w,h))

    try:
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break
            image_aug = seq(image=frame)
            video_writer.write(image_aug)
    except Exception as e:
        print(e)

    cv2.destroyAllWindows()
    video_reader.release()
    video_writer.release()

def augment_videos(i):
    try:
        video_path = f"{main_folder_path}//{video_clip_names[clip_no]}"
        video_reader = cv2.VideoCapture(video_path)
        fps = int(video_reader.get(cv2.cv2.CAP_PROP_FPS))
        w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get fps for input video
        #print(f"FPS of {video_clip_names[clip_no]} is {fps}")  
        start = time.time()
        augment_and_save_frames(video_reader,output_folder_path,video_clip_names[clip_no],i,fps,w,h)
        end = time.time()
        print("Total time taken by single video", end-start)
    except Exception as e:
        print(e)



time_of_code = time.time()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-folder-path', type=str,default='',help='Path of folder that contains classes of clips to be augmented',required=True)
    parser.add_argument('--output-folder-path', type=str, default='',required=True,help='Path of folder that will contain augmented clips and a temporary folder for holding augmented images')
    parser.add_argument('--max-clips', type=int,required=True,help='Max number of clips to augment per input video sample. Make sure max_clips is less than difference between rotation angle')
    
    opt = parser.parse_args()
    print("Args \n",opt)
    main_folder_path=opt.main_folder_path
    output_folder_path=opt.output_folder_path
    no_of_clips_to_augment_per_frame = opt.max_clips

    print("Output folder path", output_folder_path)
    print("Main folder path", main_folder_path)
    print("Max augmented clips", no_of_clips_to_augment_per_frame)
        
    if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path,exist_ok=True)
    
    video_clip_names = os.listdir(main_folder_path)
    print(f"Videos found are {video_clip_names}")
    no_of_clips_available = len(video_clip_names)

    # Run for each clip that needs to be augmented
    for clip_no in range(no_of_clips_available):
        # Rotate the clip based on angle range and increment the subsequent clips w.r.t. the angle increment
        print("No. of videos to be augmented per input", no_of_clips_to_augment_per_frame)
        with concurrent.futures.ThreadPoolExecutor() as executor:        
            executor.map(augment_videos, list(range(no_of_clips_to_augment_per_frame)))

    

    end_time = time.time()
    print("Full time by code", end_time-time_of_code)