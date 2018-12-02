import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('car.jpg')

# Initiate STAR detector
orb = cv2.ORB()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)


cv2.imshow('ORB ',img2)
cv2.waitKey()
cv2.destroyAllWindows()



