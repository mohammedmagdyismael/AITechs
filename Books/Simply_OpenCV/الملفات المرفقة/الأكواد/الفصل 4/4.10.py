import numpy as np
import cv2
img = cv2.imread('hand.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,230,255,0)
thresh_inv=cv2.bitwise_not(thresh)

contours, hierarchy = cv2.findContours(thresh_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt =contours[0]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
print defects
print defects.shape[0] 
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)

#cv2.drawContours(img, contours[0], -1, (0,0, 255), 3)

cv2.imshow('Image_hull',img)
cv2.waitKey(0)
cv2.destroyAllWindows()






