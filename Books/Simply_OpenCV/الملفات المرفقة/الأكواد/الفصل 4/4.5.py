import numpy as np
import cv2
img = cv2.imread('hand.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,245,255,0)
thresh_inv=cv2.bitwise_not(thresh)

contours, hierarchy = cv2.findContours(thresh_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt =contours[0]
hull = cv2.convexHull(cnt)

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#cv2.drawContours(img, contours[0], -1, (0,0, 255), 3)

cv2.imshow('Image_hull',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


