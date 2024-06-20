import cv2 as cv
import matplotlib as mplot
import numpy as np

alg="haarcascade_frontalface_default.xml"

haar_cascade=cv.CascadeClassifier(alg)

cam=cv.VideoCapture(0)

while True:
    _,img = cam.read()
    text="Face not detected"
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        text="Face Detected"
        cv.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
    print(text)
    image = cv.putText(img, text, (50,50), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv.LINE_AA)
    cv.imshow("Face Detection", image)
    key = cv.waitKey(10)
    if key == 27:
        break

cam.release()
cv.destroyAllWindows()
