import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose image to detect
#img = cv2.imread('pp.jpg')

#To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    #Must convert img to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #print(face_coordinates)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 10)

    
    #Display the images
    cv2.imshow('Ayush Soni', frame)
    key=cv2.waitKey(1)

    ##Stop if Q is pressed
    if key==81 or key==113:
        break
#Release the VideoCapture Object
webcam.release()

    

"""



#Display the images
cv2.imshow('Ayush Soni', img)
cv2.waitKey()
"""
print("Code Completed")
