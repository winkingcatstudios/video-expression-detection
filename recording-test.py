import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

face_cascade = cv2.CascadeClassifier('C:\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

model = model_from_json(open("models-and-weights/facial_expression_model_structure.json", "r").read())
model.load_weights('models-and-weights/facial_expression_model_weights.h5') #load weights
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# vars to log emotion counts
emotion_count, angry, disgust, fear, happy, sad, surprise, neutral = 0, 0, 0, 0, 0, 0, 0, 0

# Comment out 1 vid_path line below for test
#vid_path = "vids/exp_vid_multHead_test_1.mp4"
#vid_path = "vids/exp_vid_test_1.mp4"
#vid_path = "vids/exp_vid_zoom_test_1.mp4.mp4"
vid_path ="vids/mini_zoom_test.mp4"

# Text file for output
f = open("output/emotional_analysis_output.txt", 'w')
f.write("Output for test video: {}\n".format(vid_path))

# Define a video capture object, choose one below
vid = cv2.VideoCapture(vid_path)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frameRaw = vid.read()
    
    # mirror cam (optional) If unused, comment out and rename vid read line "frame"
    frame = cv2.flip(frameRaw,1)

    # Deepface begin
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image

        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

        img_pixels = image.img_to_array(detected_face)
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


        #write emotion text above rectangle
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Deepface end

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
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