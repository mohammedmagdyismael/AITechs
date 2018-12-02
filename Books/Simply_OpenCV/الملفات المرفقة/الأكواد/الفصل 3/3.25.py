import cv2
import numpy as np

img = cv2.imread('sonic.jpg',1)
lower_reso = cv2.pyrDown(img)

cv2.imshow('orginal',img)
cv2.imshow('lower_reso Image',lower_reso)
k = cv2.waitKey(0)
cv2.destroyAllWindows()



