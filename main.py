import cv2
import face_recognition
import math
import numpy as np
from datetime import datetime
import json

def timeLog(name):
    with open("D:\\university\\programming\\python\\img\\log.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            data = line.split(",")
            nameList.append(data[0])
        
        if name not in nameList:
            now = datetime.now()
            timeSting = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name}:{timeSting}")
            
def face_confidence(face_distance, face_match_threshold = 0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2)
    
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100), 2) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2))) * 100
        return str(round(value, 2)) + "%"
    
#import encoded dataset
with open('EncodeFiles.json', 'r') as file:
    encodeListWithNames = json.load(file)

encodeList = encodeListWithNames['encodings']
personNames = encodeListWithNames['names']
print("Đã cập nhật khuôn mặt")
print(len(encodeList), len(personNames))
#webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
    
    
    facecurrFrame = face_recognition.face_locations(frameS)
    encodeCurrFrame = face_recognition.face_encodings(frameS, facecurrFrame)
    
    for encodeFace, faceLocation in zip(encodeCurrFrame, facecurrFrame):
        #encodeFace = np.expand_dims(encodeFace, axis=0)
        name = "unknown"
        confidence = "unknown"
        match = face_recognition.compare_faces(encodeList, encodeFace)
        faceDiff = face_recognition.face_distance(encodeList, encodeFace)
        matchIndex = np.argmin(faceDiff)
        
        '''if(faceDiff[matchIndex] < 0.5):
            name = personNames[matchIndex].upper()
            timeLog(name)
        else:
            name = "unknown"'''
        
        if match[matchIndex]:
            name = personNames[matchIndex].upper()
            confidence = face_confidence(faceDiff[matchIndex])
        
        y1, x2 ,y2, x1 = faceLocation
        y1, x2 ,y2, x1 = y1*4, x2*4 ,y2*4, x1*4
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {confidence}", (x2,y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Nhat", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
