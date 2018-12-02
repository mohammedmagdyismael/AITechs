import cv2
import numpy as np

img = cv2.imread('car.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)

cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()


