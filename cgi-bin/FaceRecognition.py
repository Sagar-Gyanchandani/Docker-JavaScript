import cv2
import numpy as np
import os
import smtplib
import cgi



print("content-type: text/html")


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = cv2.face_LBPHFaceRecognizer.create()
model.read('model_saved.yml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
#         print(results[0])
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        
        
        if results[0]==1:
            if confidence > 80:
                cv2.putText(image, "Hello, Sagar", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image, display_string, (0, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,255,250), 2)
                cv2.imshow('Face Recognition', image )
                
        elif(results[0]==2):
            if confidence > 80:
                cv2.putText(image, "Hello, Rishabh", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image, display_string, (0, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,255,250), 2)
                cv2.imshow('Face Recognition', image )
            
#             sendEmail()
#             os.system("chrome aws ec2 run-instances --image-id ami-0ad704c126371a549 --instance-type t2.micro --count 1 --subnet-id subnet-a35fb8c8  --tag-specifications=[{Key=Name,Value=first_cli}] --security-group-ids sg-0a821d02796fab09b --key-name aws_key_pair")
#             os.system("firefox https://mail.google.com/mail/u/0/#inbox")
#             os.system("notepad")
#             os.system("wmplayer   c:\lw.mp3")
#             break
         
        else: 
            cv2.putText(image, "Unknown Person", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "Unknown Person", (220, 120) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Looking for Face", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
   

cap.release()
cv2.destroyAllWindows()  
   
