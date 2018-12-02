import cv2
import numpy as np
img = cv2.imread('gift.jpg')

res = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

print img.shape
print res.shape


