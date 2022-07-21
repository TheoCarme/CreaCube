import numpy as np
import cv2 as cv

from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

class Hands_Tracker():
    def __init__(self, num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super(Hands_Tracker, self).__init__()

        self.tracker = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=num_hands,
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)
        
        self.hands_results = None
        

        
    def update(self, image) :
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.hands_results = self.tracker.process(image)            
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)



    def draw_and_export_csv(self, image, csv_write, height, width) :
    
    
        def landmarks_list_to_array(landmark_list, image_shape):
          rows, cols, _ = image_shape
          return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z)
                             for lmk in landmark_list.landmark])
        
        if csv_write :
            hands_landmarks = np.full((4, 63), None)
            hands_sides = np.full((6), None)
            hands_scores = np.full((6), None)
        else :
            hands_landmarks = None
            hands_sides = None
            hands_scores = None
        
        # Draw the hand annotations on the image.            
        if self.hands_results.multi_hand_landmarks and self.hands_results.multi_handedness :
            
            min_id = []                    
        
            if csv_write :
                for idx, hand_handedness in enumerate(self.hands_results.multi_handedness):
                    hands_sides[idx] = hand_handedness.classification[0].label
                    hands_scores[idx] = hand_handedness.classification[0].score
                                    
                if np.count_nonzero(hands_scores) > 5 :
                    min_id.append(np.argmin(hands_scores[:6]))
                    np.delete(hands_scores, min_id)
                    np.delete(hands_sides, min_id) 
                    
                if np.count_nonzero(hands_scores) > 4 :
                    min_id.append(np.argmin(hands_scores[:5]))
                    np.delete(hands_scores, min_id)
                    np.delete(hands_sides, min_id)                          
                
            id = 0
            for landmarks in self.hands_results.multi_hand_landmarks :
                if id not in min_id :
                    mp_drawing.draw_landmarks(image,
                                              landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())
                
                    if csv_write :
                        array_hand_landmarks = landmarks_list_to_array(landmarks,\
                                                                      (width,  height, 3))
                        hands_landmarks[id] = array_hand_landmarks.flatten()
                    id += 1
                    
        return hands_landmarks, hands_sides, hands_scores