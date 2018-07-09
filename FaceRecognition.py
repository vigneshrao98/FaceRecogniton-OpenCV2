#Opencv2 Face, eyeFrames and smile detector using Haar feartures

import cv2

#Loading cascades, available on github: opencv/opencv/data/haarcascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# 
def detectFeatures(greyScale, rgbScale):
    faceFrames = faceCascade.detectMultiScale(greyScale, 1.3, 5)
    for (x, y, w, h) in faceFrames:
        cv2.rectangle(rgbScale, (x, y), (x+w, y+h), (255, 0, 0), 2)
        localFaceGrey = greyScale[y:y+h, x:x+w]
        localFaceRGB = rgbScale[y:y+h, x:x+w]
        eyeFrames = eyeCascade.detectMultiScale(localFaceGrey, 1.1, 22)
        
        for (x1, y1, w1, h1) in eyeFrames:
            cv2.rectangle(localFaceRGB, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
        
        smileFrames = smileCascade.detectMultiScale(localFaceGrey, 1.7, 22)
        for (x2, y2, w2, h2) in smileFrames:
            cv2.rectangle(localFaceRGB, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
    return rgbScale

#Face Recognition using detectFeatures function(needs webcam)
video = cv2.VideoCapture(0)
while True:
    _, rgbScale = video.read()
    greyScale = cv2.cvtColor(rgbScale, cv2.COLOR_BGR2GRAY)
    detected = detectFeatures(greyScale, rgbScale)
    cv2.imshow('Video', detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()