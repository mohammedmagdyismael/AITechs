import cv2
import numpy as np

img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))  #stacking images side-by-side

cv2.imshow('res.png',res)
cv2.waitKey(0)
cv2.destroyAllWindows()






