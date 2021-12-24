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