import cv2
import numpy as np
import pandas as pd
import time
from keras.preprocessing import image as kerasImage
from keras.models import model_from_json

detector = cv2.dnn.readNetFromCaffe("models-and-weights/deploy.prototxt" , "models-and-weights/res10_300x300_ssd_iter_140000.caffemodel")

model = model_from_json(open("models-and-weights/facial_expression_model_structure.json", "r").read())
model.load_weights('models-and-weights/facial_expression_model_weights.h5') #load weights
emotions = ('angry', 'disgust', 'thinking', 'happy', 'sad', 'surprise', 'neutral') # Changed fear to thinking

# vars to log emotion counts
emotion_count, angry, disgust, fear, happy, sad, surprise, neutral = 0, 0, 0, 0, 0, 0, 0, 0

# Define a video capture object, choose one below
vid = cv2.VideoCapture(0)


start = time.time()
current_time = time.time()+1
last_emotion = emotions[0]

while(True):
      
    # Capture the video frame
    # by frame
    ret, image = vid.read()
    
    # Deepface begin
    if(ret):
        # Manip frame: reduce, flip, recolor
        base_img = image.copy()
        original_size = image.shape
        target_size = (300, 300)
        image = cv2.resize(image, target_size)
        aspect_ratio_x = (original_size[1] / target_size[1])
        aspect_ratio_y = (original_size[0] / target_size[0])

        # Get faces
        imageBlob = cv2.dnn.blobFromImage(image = image)
        
        detector.setInput(imageBlob)
        detections = detector.forward()

        detections_df = pd.DataFrame(detections[0][0]
            , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])

        detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
        detections_df = detections_df[detections_df['confidence'] >= 0.50]  # default: 0.90


        for i, instance in detections_df.iterrows():
            # Print(instance)
            
            confidence_score = str(round(100*instance["confidence"], 2))+" %"
            
            left = int(instance["left"] * 300)
            bottom = int(instance["bottom"] * 300)
            right = int(instance["right"] * 300)
            top = int(instance["top"] * 300)
            
            # High resolution
            detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
            
            if detected_face.any():
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) # Transform to gray scale
                detected_face = cv2.resize(detected_face, (48, 48)) # Resize to 48x48

                img_pixels = kerasImage.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)

                img_pixels /= 255 # Pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

                predictions = model.predict(img_pixels) # Store probabilities of 7 expressions

                # Find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                max_index = np.argmax(predictions[0])

                current_time = time.time()
                if current_time - start >= 0.5:
                    emotion = emotions[max_index]
                    last_emotion = emotions[max_index]
                    start = current_time
                else:
                    emotion = last_emotion

                if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:

                    # High resolution
                    cv2.putText(base_img, emotion, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(bottom*aspect_ratio_y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255),1) #draw rectangle to main image

        # Display the resulting frame
        cv2.imshow('frame', base_img)
        
        # The 'q' button quits
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()