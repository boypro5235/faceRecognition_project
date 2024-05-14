import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import json

def MaHoa(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgFaceLocation = face_recognition.face_locations(img)
        encode = face_recognition.face_encodings(img, imgFaceLocation)[0]
        encodelist.append(encode)
    return encodelist

#training data    
path = 'D:\\university\\programming\\python\\img\\dataset'
images = []
personNames = []

List = os.listdir(path)
for folder in List:
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):  
        personNames.append(folder.title())
        image_file = os.listdir(folder_path)[0]  
        image_path = os.path.join(folder_path, image_file)
        curImg = cv2.imread(image_path)
        images.append(curImg)

print("Bắt đầu mã hóa")    
encodeList = MaHoa(images)
encodeList = [encode.tolist() for encode in encodeList]
encodeListWithNames = {'encodings': encodeList, 'names': personNames}
print("Mã hóa thành công")

# Lưu danh sách mã hóa và tên vào tệp JSON
with open("EncodeFiles.json", 'w') as json_file:
    json.dump(encodeListWithNames, json_file)

print("Đã lưu")
