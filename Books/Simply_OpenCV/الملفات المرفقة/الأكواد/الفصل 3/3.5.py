import cv2
import numpy as np
def nothing(x):
   pass
img = cv2.imread('shape2.jpg',0)
#Generate trackbar Window Name
TrackbarName='value threshold'

#Make Window and Trackbar
cv2.namedWindow('Window')
cv2.createTrackbar(TrackbarName,'Window',0,255,nothing)

# Allocate destination image
Threshold = np.zeros(img.shape, np.uint8)

# Loop for get trackbar pos and process it
while True:
   #Get position in trackbar
   TrackbarPos = cv2.getTrackbarPos(TrackbarName,'Window')

   #Apply Threshold
   cv2.threshold(img,TrackbarPos,255,cv2.THRESH_BINARY, Threshold)

   # Show in window
   cv2.imshow('Window',Threshold)

   # If you press "ESC", it will return value
   ch = cv2.waitKey(5)
   if ch == 27:
      break

cv2.destroyAllWindows()

