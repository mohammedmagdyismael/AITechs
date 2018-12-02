import cv2
import numpy as np

img = cv2.imread('hello.png',0)
rows,cols = img.shape
pts1 = np.float32([[67,160],[370,115],[371,263]])
pts2 = np.float32([[20,114],[370,115],[371,263]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('orginal',img)
cv2.imshow('res',dst)

k = cv2.waitKey(0)

cv2.destroyAllWindows()



