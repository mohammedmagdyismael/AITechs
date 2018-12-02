import numpy as np
import cv2

img =cv2.imread('brid.jpg')

# Draw circle
cv2.circle(img,(480,120), 80, (0,0,255), 5)

cv2.imshow('imag',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
