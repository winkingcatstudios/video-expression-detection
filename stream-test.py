import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

face_cascade = cv2.CascadeClassifier('C:\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

model = model_from_json(open("models-and-weights/facial_expression_model_structure.json", "r").read())
model.load_weights('models-and-weights/facial_expression_model_weights.h5') #load weights
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frameRaw = vid.read()
    
    # mirror cam (optional) If unused, comment out and rename vid read line "frame"
    frame = cv2.flip(frameRaw,1)

    # Deepface begin
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
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()