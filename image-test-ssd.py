import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

detector = cv2.dnn.readNetFromCaffe("models-and-weights/deploy.prototxt" , "models-and-weights/res10_300x300_ssd_iter_140000.caffemodel")

image = cv2.imread("pics/zoom_randos.jpg") #zoom_test_pic.PNG AND zoom_randos.jpg
base_img = image.copy()
original_size = image.shape
target_size = (300, 300)
print("original image size: ", original_size)

image = cv2.resize(image, target_size)
aspect_ratio_x = (original_size[1] / target_size[1])
aspect_ratio_y = (original_size[0] / target_size[0])
print("aspect ratios x: ",aspect_ratio_x,", y: ", aspect_ratio_y)
print(image.shape)

plt.imshow(image[:,:,::-1])
plt.waitforbuttonpress()

#detector expects (1, 3, 300, 300) shaped input
imageBlob = cv2.dnn.blobFromImage(image = image)
#imageBlob = np.expand_dims(np.rollaxis(image, 2, 0), axis = 0)

detector.setInput(imageBlob)
detections = detector.forward()

print(detections[0][0].shape)

detections_df = pd.DataFrame(detections[0][0]
    , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])

detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
detections_df = detections_df[detections_df['confidence'] >= 0.50]  # default: 0.90

print(detections_df.head())

for i, instance in detections_df.iterrows():
    #print(instance)
    
    confidence_score = str(round(100*instance["confidence"], 2))+" %"
    
    left = int(instance["left"] * 300)
    bottom = int(instance["bottom"] * 300)
    right = int(instance["right"] * 300)
    top = int(instance["top"] * 300)
        
    #low resolution
    #detected_face = image[top:bottom, left:right]
    
    #high resolution
    detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
    
    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
        
        #plt.figure(figsize = (3, 3))
        
        #low resolution
        #cv2.putText(image, confidence_score, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 1) #draw rectangle to main image
        
        #high resolution
        cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image
        
        #-------------------
        
        print("Id ",i)
        print("Confidence: ", confidence_score)
        #detected_face = cv2.resize(detected_face, (224, 224))
        plt.imshow(detected_face[:,:,::-1])
        plt.axis('off')
        plt.waitforbuttonpress()


plt.figure(figsize = (20, 20))
#tmp_img = image.copy()
#tmp_img = cv2.resize(tmp_img, (600, 600))
plt.imshow(base_img[:,:,::-1])
plt.axis('off')
plt.waitforbuttonpress()