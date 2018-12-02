import numpy as np
import cv2

img1 = cv2.imread('carr.png') # queryImage
img2 = cv2.imread('car.jpg') # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()
    
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_LSH = 6
#help(dict)
index_params= dict(algorithm = FLANN_INDEX_LSH,
           table_number = 6, # 12
           key_size = 12, # 20
           multi_probe_level = 1) #2

search_params = dict(checks=50) # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
    singlePointColor = (255,0,0),
    matchesMask = matchesMask,
    flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

cv2.imshow('MatchesKnn ',img3)
cv2.waitKey()
cv2.destroyAllWindows()


