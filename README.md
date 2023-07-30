
# Word Level Indian Sign Language Recognition 

This project aims to bridge the communication gap for the deaf community by providing them with a mobile/web app that predicts the meaning of sign gestures captured in a video, allowing them to interact with non-sign language users effectively.


#### Brief Explanation:
- This project is aimed at developing a word-level Indian Sign Language (ISL) recognition system on Videos. Sign language is an action word which cannot be encompassed in a single frame, so a video of 2-3 seconds is required to determine the word signified by the action. The project's primary objective is to bridge the communication gap for the deaf community by enabling them to interact with non-sign language users effectively. The code sample provided here showcases the functionality of the system, demonstrating how it processes video input to recognize and predict the words signified by specific sign language actions.

- The code begins by accepting a video feed, which can be obtained from various sources such as a camera, a saved video file, or a web application. From this feed, the system selects 45 frames evenly to ensure it doesn't lose information if number of frames is greater than 45 in videos. Using Mediapipe PoseNet, an advanced pose-estimation model, the code detects and captures key body, left hand, and right-hand coordinates present in each frame.For example, movements of the right thumb across frames can help identify specific signs.

- These extracted coordinates are then saved as a Numpy array, preserving the spatial information of the key points. Shape: (45 frames, 22 coordinate information in x and y axis) Subsequently, the Numpy array is fed into an LSTM (Long Short-Term Memory) deep learning model. The LSTM model is crucial for processing sequential data and understanding the temporal dynamics of sign language gestures.

- Finally, the LSTM model predicts the word corresponding to the sign language action in the video. This predicted word is the system's output, representing the meaningful interpretation of the sign language gesture


### Output Words: [Hello, How are you, Thank you]


## Project WorkFlow :
![image](https://github.com/Sooryak12/ISL_Recognition/assets/55055042/d9312bf1-d615-4fc0-a2d3-edf3b4d08709)


## Installation and Usage

1. Install Necessary Libraries: 

```bash
pip install -r requirements.txt
```

2. Create a web-app:

```python
python app.py
```


3. Run with laptop camera (live feed):

```python
python deploy_code.py
```

4. Process saved video from the command line:

```python
python run_through_cmd_line.py -i input_file_path
```


## API Reference

#### Upload Video in form:

```http
POST /upload-video/
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file`    | `file`   | Video                      |

#### Test API call:

```http
GET /test/
```


## Data 

Data Source: 
1. [INCLUDE 50 Dataset](https://zenodo.org/record/4010759)
2. Video collection by recording ourselves doing the actions.

- The INCLUDE 50 dataset contains 50 words grouped under tags like greetings, pronouns, and each word contains 25 videos with augmentations.
- Since 25 videos are not enough to train a deep learning model, we recorded additional data by capturing videos of the actions ourselves.
- We added a 'None' class to signify no action in the video to prevent random outputs when no action is performed.
- Our final test set includes 20 videos of our friends performing the actions in real-time scenarios.
- Initially, we experimented with a CRNN model (V1) trained with 16 videos. Although the model showed considerable F1 score, it performed poorly in real-time and suffered from underfitting.
- Later, we limited the classes and collected more data. The LSTM model was trained with 3 words: [Hello, How are you, Thank You], each containing 100 videos.

#### Dataset Details :

##### Actions: [Hello, How are you, Thank you]
Training Data : 
 | Include Dataset Videos| Own Videos  | Number of Videos after augmentation                | Total Training Data |
| :-------- | :------- | :------------------------- | :----------|
| 25 videos per action | 60 Videos Per action | 340 Videos per action | 1020 Videos

Validation  Data : 
 | Include Dataset Videos| Own Videos  |  Total Validation  Data |
| :-------- | :------- | :------------------------- |
| 10 videos per action | 20 Videos Per action | 90 Videos  | 

Real Time Test Data : 

| Videos per action| Total   Data |
| :-------- | :------- | 
| 8 Videos  | 24 Videos  | 




## Input Shape

We experimented with various mobile devices to detect the number of frames captured per second and decided that 45 frames would provide sufficient information for predicting a class without losing crucial details.

```
If there are more than 45 frames in the captured video:
     We evenly select frames to avoid losing information.
Else if the number of frames is less than 45:
     We add empty frames to maintain a constant input shape.
```


## Mediapipe PoseNet Detection 

Mediapipe PoseNet for Sign Language Project:
Mediapipe PoseNet, a real-time pose estimation model, lies at the core of our sign language project. Leveraging PoseNet's robustness, we achieve precise tracking of body keypoints, enabling accurate interpretation of sign gestures. With seamless integration and customization options, our application empowers effective communication for the deaf and hard of hearing community.


## Architectures

```
V1: CRNN Model Architecture
   - A Time Distributed MobileNet model takes keypoints embedded in video as input.
   - These are passed through LSTMs and Dense layers to classify actions.

V2: LSTM Architecture (Final Version)
   - Numpy array containing keypoint coordinates in each frame as a separate array. Array Shape: (45, 24, 2)
   - These are fed through LSTMs to classify the actions.
```
## Results 

LSTM Model performed significantly well both in validation set and exceptionally well in real time test data.

Categorical Accuracy  : 
 | Train | Validation   | Real-Time Test Data                |
| :-------- | :------- | :------------------------- |
| 78   % | 74.6 % | 95 %                     |


CRNN Time Distributed Model (V1) Categorical Accuracy:

 | Train | Validation   | Real-Time Test Data                |
| :-------- | :------- | :------------------------- |
| 82%    | 42.4%  | 5%              |


Model Size Comparision :

 | CRNN Model (V1) | LSTM Model   | 
| :-------- | :------- | 
| 323mb (saved weights and structure)| 2.3mb (only weights)| 



## Screenshots 


1. Web app :

![ApplicationFrameHost_BbTrmb1MGc](https://github.com/Sooryak12/ISL_Recognition/assets/55055042/7facb461-18b7-4eb2-adca-2161a9bce712)

2. Run with Laptop camera as feed.
![ApplicationFrameHost_H5ia1qMcXq](https://github.com/Sooryak12/ISL_Recognition/assets/55055042/476e0c90-728f-44e1-9b9a-bd127695dbd1)

## Background

- Sign language is used by members of the deaf community for communication, where each hand gesture corresponds to a specific meaning.
- In India, there are over 5 million deaf individuals, but there are only 250 certified sign language interpreters, resulting in one interpreter for every 20,000 deaf people.
- To address this imbalance, we propose the "Word Level Sign Language Recognizer" for Indian Sign Language, using the INCLUDE Dataset containing 2-3 second videos with corresponding signs.
- This project can be a game-changer, enabling deaf community members to interact more easily with others.
- Our approach involves extracting key pose feature points (body positions) and using a neural network architecture to find spatial differences between frames, allowing us to build a model for classifying signs into words.


## Acknowledgements

- [INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition](https://dl.acm.org/doi/10.1145/3394171.3413528)
- [Nicholas Ronette Tutorials](https://www.youtube.com/@NicholasRenotte)



