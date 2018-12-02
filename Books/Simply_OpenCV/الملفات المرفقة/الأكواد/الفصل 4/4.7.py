import numpy as np
import cv2
img = cv2.imread('airplan.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,207,255,0)
thresh_inv=cv2.bitwise_not(thresh)

kernel = np.ones((15,15),np.uint8)
dilation = cv2.dilate(thresh,kernel,iterations = 1)
cv2.imshow('dilation',dilation)

contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt =contours[1]

#draw hull Rectangle 
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('Image_hull',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



