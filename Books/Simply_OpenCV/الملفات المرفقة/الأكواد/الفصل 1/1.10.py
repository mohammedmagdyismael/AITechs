import cv2
import numpy as np

drawing = False
mode = False
ix,iy = -1,-1
# mouseCallback Function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,spareimg,img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                img = spareimg.copy()
                spareimg = img.copy()
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
            else:
                cv2.circle(img,(ix,iy),3,(0,0,255),-1)
                cv2.line(img,(ix,iy),(x,y),(0,0,255),3)
                ix,iy = x,y
         
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True :
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
            spareimg = img.copy()
        else:
            cv2.circle(img,(x,y),3,(0,0,255),-1)

img = cv2.imread('gift.jpg',1)
spareimg = img.copy()
# img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1)
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    
cv2.destroyAllWindows()

