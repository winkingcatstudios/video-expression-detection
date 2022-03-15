import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image as kerasImage
from keras.models import model_from_json

vid_path = ""
emotion_count, angry, disgust, fear, happy, sad, surprise, neutral = 0, 0, 0, 0, 0, 0, 0, 0
dom_emo_string = ""

def set_vid_path(path):
     global vid_path
     vid_path = path

def run():
    detector = cv2.dnn.readNetFromCaffe("models-and-weights/deploy.prototxt" , "models-and-weights/res10_300x300_ssd_iter_140000.caffemodel")

    model = model_from_json(open("models-and-weights/facial_expression_model_structure.json", "r").read())
    model.load_weights('models-and-weights/facial_expression_model_weights.h5') #load weights
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    # vars to log emotion counts
    global emotion_count
    global angry
    global disgust
    global fear
    global happy
    global sad
    global surprise
    global neutral
    global dom_emo_string
    # emotion_count, angry, disgust, fear, happy, sad, surprise, neutral = 0, 0, 0, 0, 0, 0, 0, 0

    # Text file for output
    f = open("output/emotional_analysis_output.txt", 'w')
    f.write("Output for test video: {}\n".format(vid_path))

    # Define a video capture object, choose one below
    vid = cv2.VideoCapture(vid_path)

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
                #print(instance)
                
                confidence_score = str(round(100*instance["confidence"], 2))+" %"
                
                left = int(instance["left"] * 300)
                bottom = int(instance["bottom"] * 300)
                right = int(instance["right"] * 300)
                top = int(instance["top"] * 300)
                
                #high resolution
                detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

                img_pixels = kerasImage.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)

                img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

                predictions = model.predict(img_pixels) #store probabilities of 7 expressions

                #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                max_index = np.argmax(predictions[0])

                emotion = emotions[max_index]

                # log max emotion
                emotion_count += 1
                if(emotion == "angry"):
                    angry += 1
                elif(emotion == "disgust"):
                    disgust +=1
                elif(emotion == "fear"):
                    fear +=1
                elif(emotion == "happy"):
                    happy +=1
                elif(emotion == "sad"):
                    sad +=1
                elif(emotion == "surprise"):
                    surprise +=1
                elif(emotion == "neutral"):
                    neutral +=1

                if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:

                    #high resolution
                    cv2.putText(base_img, emotion, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(bottom*aspect_ratio_y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255,0,0),2) #draw rectangle to main image

            # Display the resulting frame
            cv2.imshow('frame', base_img)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    # Write emotion log output and close file
    f.write("Emotions Counted: {}\n".format(emotion_count))
    f.write("---Angry---\nCount: {}\nPercent:{}\n\n".format(angry, angry/emotion_count*100))
    f.write("---Disgust---\nCount: {}\nPercent:{}\n\n".format(disgust, disgust/emotion_count*100))
    f.write("---Fear---\nCount: {}\nPercent:{}\n\n".format(fear, fear/emotion_count*100))
    f.write("---Happy---\nCount: {}\nPercent:{}\n\n".format(happy, happy/emotion_count*100))
    f.write("---Sad---\nCount: {}\nPercent:{}\n\n".format(sad, sad/emotion_count*100))
    f.write("---Surprise---\nCount: {}\nPercent:{}\n\n".format(surprise, surprise/emotion_count*100))
    f.write("---Neutral---\nCount: {}\nPercent:{}\n\n".format(neutral, neutral/emotion_count*100))

    # log max emotion
    dom_emo = max(angry, disgust, fear, happy, sad, surprise, neutral)
    if(dom_emo == angry):
        dom_emo_string = "Angry"
    elif(dom_emo == disgust):
        dom_emo_string = "Disgust"
    elif(dom_emo == fear):
        dom_emo_string = "Fear"
    elif(dom_emo == happy):
        dom_emo_string = "Happy"
    elif(dom_emo == sad):
        dom_emo_string = "Sad"
    elif(dom_emo == surprise):
        dom_emo_string = "Surprise"
    elif(dom_emo == neutral):
        dom_emo_string = "Neutral"

    f.write("Dominant Emotion: {}\n".format(dom_emo_string))
    f.close()
