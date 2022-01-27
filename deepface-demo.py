from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json

img_path = "pics/shan2.jpg"
img = cv2.imread(img_path)
plt.imshow(img[:,:,::-1])
plt.show()

dict_res = DeepFace.analyze(img_path = img_path, actions = ['age', 'gender', 'race', 'emotion'])
string_res = json.dumps(dict_res)
json_res = json.loads(string_res)
age_res = json_res["age"]
print(age_res)