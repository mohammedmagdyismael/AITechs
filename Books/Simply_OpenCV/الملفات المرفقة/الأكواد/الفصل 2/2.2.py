import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('logo_opencv.png',1)
img2 = cv2.imread('football.jpg',1)

# 600X600 sound cool
# img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
img2 = cv2.resize(img2,(750,600))
img1 = cv2.resize(img1,(750,600))

res = cv2.addWeighted(img1,0.7,img2,0.3,0)

plt.imshow(res)
plt.xticks([]),plt.yticks([])
plt.show()

