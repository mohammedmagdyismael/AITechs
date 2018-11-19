# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 01:46:15 2018

@author: mohammed-PC
"""

'''
Modules Imports
'''
import os
import csv
import cv2
import sys 
import dlib
import time
import keras
import imutils
import platform
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot as plt


class ID:
    def __init__(self, frontSide, backSide):
        try:
            '''Display Environment Elements'''
            print("## Environment")
            print("OS: "+platform.system()+" "+platform.release()+" x"+platform.architecture()[0])
            print("Keras Version:",keras.__version__)
            print("TensorFlow Version:",tf.__version__)
            print("Opencv Version:",cv2.__version__)
            
            '''Variables and Instances'''
            self.side = ''
            self.file_name = 'data_store.csv'
            self.classifier = load_model('./assets/ar_numbers_v6.h5')
            self.classifier.compile(loss= 'categorical_crossentropy', optimizer='adam')
            self.detector = dlib.get_frontal_face_detector()
            
            '''Assets''' 
            self.assets = './assets/images/'
            self.temp_egy_logo = cv2.imread(self.assets+'logo_0.jpg')
            self.temp_nesr = cv2.imread(self.assets+'logo_1.jpg')
            self.temp_pharaonic = cv2.imread(self.assets+'logo_2.jpg')
            
            '''Load both sides of ID card and display their resolution'''
            dataset ='./dataset/' 
            self.back_img = cv2.imread(dataset+ backSide )
            self.face_img = cv2.imread(dataset+ frontSide )
            print("Front Side Resolution: ", self.face_img.shape )
            print("Back Side Resolution: ", self.back_img.shape )
            
            '''Classifier Input Shape'''
            self.NIDimage = (64,64)
            self.clsfInputWidth = self.NIDimage[0]
            self.clsfInputHeight = self.NIDimage[1]
            
            print("\nProject Variables and Instances are Loaded successfully")
        except:
            '''If any of the System variables or 
               instances didn't load, the system 
               terminates
            '''
            print("Some Instances Can't Load !")
            sys.exit()
    
    
    def crop_points(self, image):
        """
        This Function displays an Interactive UI enables user
        to select four corner points, then return new rectangular
        prespective
        
        Args:
            image (3D Array): RGB Image
        
        Returns:
            image (3D Array): RGB Image with new prespectives
            """
        new_image = image.copy()
        drawing=False
        mode=True
        points_presp = []
        def setPrespectives(image_file , arr ):
             def order_points(pts):
                 """
                 This Function orders four points to shape a rectangle
                 
                 Args:
                     pts (2D List): 4 points
                     
                 Returns:
                    rect (2D List): recatnge ordered 4 corner points
                 """
                 rect = np.zeros((4, 2), dtype = "float32")
                 s = pts.sum(axis = 1)
                 rect[0] = pts[np.argmin(s)]
                 rect[2] = pts[np.argmax(s)] 
                 diff = np.diff(pts, axis = 1)
                 rect[1] = pts[np.argmin(diff)]
                 rect[3] = pts[np.argmax(diff)]
                 return rect
        
             def four_point_transform(image, pts):
                 rect = order_points(pts)
                 (tl, tr, br, bl) = rect
                 widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                 widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                 maxWidth = max(int(widthA), int(widthB))
                 heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                 heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                 maxHeight = max(int(heightA), int(heightB))
                 dst = np.array([
                                 [0, 0],
                                 [maxWidth - 1, 0],
                                 [maxWidth - 1, maxHeight - 1],
                                 [0, maxHeight - 1]], dtype = "float32")
                 M = cv2.getPerspectiveTransform(rect, dst)
                 warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                 return warped
             
             img = image_file
             arr = arr
             arr = np.array(arr, dtype = "float32")
             warped = four_point_transform(img, arr)
             h , w , l = warped.shape
             h = int( w/1.5 )
             warped = cv2.resize(warped , (w , h))
             return warped
         
        def interactive_drawing(event,x,y,flags,param):
             global ix,iy,drawing, mode, current_Image
             if event==cv2.EVENT_LBUTTONDOWN:
                 drawing=True
                 ix,iy=x,y
                 if len(points_presp) < 4 :
                     cv2.line(image,(x,y), (x,y) , (0,0,255) , 8 )   
                     points_presp.append([x,y])
                 else:
                    print("You Can't Select More Than 4 Points ! \n")
             elif event==cv2.EVENT_LBUTTONUP:
                 drawing=False 
             return x,y
        
        #image = imutils.resize(image, width=500)
        cv2.namedWindow('Photo',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Photo',interactive_drawing)
         
        while(1):
            cv2.imshow('Photo',image)
            k=cv2.waitKey(1)&0xFF
            if k==27:    #esc --> exit
                cv2.destroyAllWindows()
                break
            
            elif k==110: #n --> new
                image = new_image.copy()
                h , w , _ = image.shape
                points_presp = []
            
            elif k==99: #c --> crop
                 if len(points_presp) == 4:
                    image = setPrespectives(image,points_presp)
                    cv2.destroyAllWindows()
                    return image
           
             
    def crop_front(self, front, scale):
        global side
        # // must removed after test 
        side = 'face'
        #Scale
        scale_factor = scale
        #Real Card width and height
        w = 8.6
        h = 5.5
        #init. card area parameters
        scaled_h = int(h*scale_factor)
        scaled_w = int(w*scale_factor)
        
        print("New Front Side Scaled Resolution: " + str(scaled_h) + ", " + str(scaled_w) + "\n")
        
        ##Front
        #X
        face_xi , face_xf = int(0.3*scale_factor) , int((0.3+2.2)*scale_factor)
        birthdate_xi , birthdate_xf =  int(0.3*scale_factor) , int((0.3+3)*scale_factor)
        pin_xi , pin_xf =  int(0.3*scale_factor) , int((0.3+3)*scale_factor)
        logo_xi , logo_xf = int((0.3+3)*scale_factor) , int(7.5*scale_factor)
        name_xi , name_xf = int((0.3+3)*scale_factor) , int(w*scale_factor)
        add_xi , add_xf = int((0.3+3)*scale_factor) , int(w*scale_factor)
        idno_xi , idno_xf = int((0.3+3+0.3)*scale_factor) , int(w*scale_factor)
        #Y
        face_yi , face_yf = int(0.4*scale_factor) , int(3.2*scale_factor)
        birthdate_yi , birthdate_yf =  int(3.7*scale_factor) , int(4.8*scale_factor)
        pin_yi , pin_yf =  int(4.8*scale_factor) , int(h*scale_factor)
        logo_yi , logo_yf = int(0*scale_factor) , int(1.2*scale_factor)
        name_yi , name_yf = int(1.3*scale_factor) , int(2.5*scale_factor)
        add_yi , add_yf = int(2.5*scale_factor) , int(3.8*scale_factor)
        idno_yi , idno_yf = int((4.1+0.15)*scale_factor) , int(4.9*scale_factor)
        card = cv2.bitwise_not(np.zeros((scaled_h,scaled_w,3), np.uint8))
        card_f = front
        card_f = cv2.resize(card_f, (scaled_w , scaled_h)) 
        
        thick = 2
        #front 
        face = card_f[face_yi:face_yf , face_xi:face_xf]
        birthdate = card_f[birthdate_yi:birthdate_yf , birthdate_xi:birthdate_xf]
        pin = card_f[pin_yi:pin_yf , pin_xi:pin_xf]
        logo = card_f[logo_yi:logo_yf , logo_xi:logo_xf]
        name = card_f[name_yi:name_yf , name_xi:name_xf]
        add = card_f[add_yi:add_yf , add_xi:add_xf]
        idno = card_f[idno_yi:idno_yf , idno_xi:idno_xf]
        front = {'face':face,'birthdate': birthdate, 'pin': pin, 'logo': logo,'name': name, 'add': add, 'idno': idno}
        return front
    
    def is_face_card(self, card):
        #ADD AKAZE ->Edit
        is_logo = self.Matcher(self.temp_egy_logo, card['logo'])
        return (is_logo and self.is_face(card['face']))
        
        
    ##AKAZE Feature matching
    def Matcher (self, originalImage, tempelateImage):    
        ###Load Original Image and Tempelate Image 
        originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        originalImage_H, originalImage_W  = originalImage.shape
        #Original Croped Area to Compare to the matching area
        originalImage_Area = originalImage_H * originalImage_W 
         
        tempelateImage = cv2.cvtColor(tempelateImage, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        tempelateImage = clahe.apply(tempelateImage)
        
        tempelateImage_H, tempelateImage_W  = tempelateImage.shape 
        ###Create AKAZE 
        akaze = cv2.AKAZE_create()
        originalImage_kp, originalImage_des = akaze.detectAndCompute (originalImage,None)
        tempelateImage_kp, tempelateImage_des = akaze.detectAndCompute(tempelateImage, None)
        ###Find Matches
        matcher = cv2.BFMatcher()
        knnMatches = matcher.knnMatch(tempelateImage_des,originalImage_des, k=2) 
        
        good = []
        for m,n in knnMatches: 
            if m.distance < 0.9*n.distance:
                good.append(m)
        
        src_pts = np.float32([ tempelateImage_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ originalImage_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
        pts = np.float32([ [0,0],[0,tempelateImage_H-1],[tempelateImage_W-1,tempelateImage_H-1],[tempelateImage_W-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        #Edit Polylines point to get perfect squared points
        if(dst[0][0][0] > dst[1][0][0]): dst[0][0][0] = dst[1][0][0]
        if(dst[0][0][0] < dst[1][0][0]): dst[1][0][0] = dst[0][0][0]
        if(dst[3][0][0] > dst[2][0][0]): dst[2][0][0] = dst[3][0][0]  
        if(dst[3][0][0] < dst[2][0][0]): dst[3][0][0] = dst[2][0][0]  
        if(dst[0][0][1] > dst[3][0][1]): dst[0][0][1] = dst[3][0][1]
        if(dst[0][0][1] < dst[3][0][1]): dst[3][0][1] = dst[0][0][1]
        if(dst[1][0][1] > dst[2][0][1]): dst[2][0][1] = dst[1][0][1]
        if(dst[1][0][1] < dst[2][0][1]): dst[1][0][1] = dst[2][0][1] 
        
        matchingContour_H = dst[1][0][1] - dst[0][0][1]
        matchingContour_W = dst[3][0][0] - dst[0][0][0]
        matchingContour_Area = matchingContour_H * matchingContour_W
        
        matchingAreaPercentage = matchingContour_Area/originalImage_Area
        
        if (matchingAreaPercentage > 0.75):
            return True
        else:
            return False    
    
    def extract_front_data(self, front_img_parts):
        id_num = self.extract(front_img_parts['idno'], 14)
        birthdate = [2] #extract(front_img_parts['birthdate'], 10)
        return {'id': id_num, 'birthdate': birthdate}


    def show_face_result(self, face_card):
        #   will be best if next static data added in config file
        print("===================Front=====================")
        #   print("IS IT REAL CARD?: " ,face_card['is_id_card'])
        #   print("BirthDate:" , face_card['birthdate'])
        print("ID NUMBER: " ,face_card['id'])  

    
    def crop_back(self, back, scale):
        global side
        # // must removed after test 
        side = 'back'
        #Scale
        scale_factor = scale
        #Real Card width and height
        w = 8.6
        h = 5.5
        #init. card area parameters
        scaled_h = int(h*scale_factor)
        scaled_w = int(w*scale_factor)
        
        print("New Back Side Scaled Resolution: " + str(scaled_h) + ", " + str(scaled_w) + "\n")

        
        ##Back
        #X
        nesr_xi , nesr_xf =  int(7*scale_factor) , int(8.3*scale_factor)
        pharaonic_xi , pharaonic_xf = int(0.3*scale_factor) , int(2*scale_factor)
        info_xi , info_xf = int(2.4*scale_factor) , int(7*scale_factor)
        expiry_xi , expiry_xf = int(2.5*scale_factor) , int(7*scale_factor)
        code_xi , code_xf = int(0*scale_factor) , int(w*scale_factor)
        #Y
        nesr_yi , nesr_yf =  int(0*scale_factor) , int(2.2*scale_factor)
        pharaonic_yi , pharaonic_yf = int(0*scale_factor) , int(2.2*scale_factor)
        info_yi , info_yf = int(0.3*scale_factor) , int(2.2*scale_factor)
        expiry_yi , expiry_yf = int(2.4*scale_factor) , int(3*scale_factor)
        code_yi , code_yf = int(3*scale_factor) , int(h*scale_factor)
        
        card = cv2.bitwise_not(np.zeros((scaled_h,scaled_w,3), np.uint8)) 
        card_b = back
        card_b = cv2.resize(card_b, (scaled_w , scaled_h)) 
        
        #back - cut process 
        nesr = card_b[nesr_yi:nesr_yf , nesr_xi:nesr_xf]    
        pharaonic = card_b[pharaonic_yi:pharaonic_yf , pharaonic_xi:pharaonic_xf ]    
        info = card_b[info_yi:info_yf , info_xi:info_xf]
        code = card_b[code_yi:code_yf , code_xi:code_xf]
        expiry = card_b[expiry_yi:expiry_yf , expiry_xi:expiry_xf] 
     
        sliced_image = self.horizontal_split(info)
        plt.imshow(info)
        plt.figure()
        
        first_line = sliced_image[0]
    
        card_id = self.vertical_split(first_line)[0]
        # expiry = vertical_split(expiry)[1]
         
        # rename this dict 
        back = {'nesr':nesr,'pharaonic': pharaonic, 'info': info, 'code': code,'expiry': expiry,  'id': card_id }
        return back
        
    def is_back_card(self, card):
        # here will add Hog code ISA to chek back card
        is_nesr = self.Matcher(self.temp_nesr, card['nesr'])
        is_pharaonic = self.Matcher(self.temp_pharaonic, card['pharaonic'])
        return (is_nesr and is_pharaonic)
        
    def extract_back_data(self, back_img):
        id_num = self.extract(back_img['id'], 14) 
        # expiry = extract(back_img['expiry'], '*') # Expiry Date
        # plt.imshow(back_img['id'])
        # plt.figure()
        return {'id_num': id_num}

    def show_back_result(self, back_data):
        print("===================Back======================")
        if back_data is not None:
            print("ID NUMBER: ", back_data['id_num'])
            #   print("Expiry Date: ", back_data['expiry']) 
        else:
            print("there is no data to extract")
        
    def is_face(self, face): 
        plt.imshow(face)
        plt.figure()
        plt.show()
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # faces.
        faces = self.detector(gray, 1)
        print("Number of faces detected: {}".format(len(faces)))  
        return len(faces) is 1

    def horizontal_split(self, image):
        h , w, l = image.shape
        image_resized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image_resized,140,200)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        rowSum = []
    
        for r in range(0,h):
            row = edges[r:r+1 , 0:w-1 ]
            x = np.count_nonzero (row)
            rowSum.append(x)
    
        cuts_pos = []
        cnt = 0
        for i in range (0 , len(rowSum)):
            if rowSum[i] == 0 and cnt == 0:
                cuts_pos.append(i)
                cnt = 1
    
            elif rowSum[i] != 0  :
                cnt = 0
    
        if cuts_pos[0] != 0:
            cuts_pos.insert(0, 0)
    
        statments = []
        for i in range (0 , len(cuts_pos)-1):
            statments.append(image[cuts_pos[i]:cuts_pos[i+1] , 0:w])
        return statments
        
    
    def vertical_split(self, image):
        grayIamge = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
        thresh_image = self.threshold(grayIamge) 
        kernel = np.ones((12,12), np.uint8)
        dilated_image = cv2.dilate(thresh_image, kernel, iterations=1)
        edges, contours, hierarchy = cv2.findContours(dilated_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        lst = []
        slices = []
    
        for i in range(0,len(contours)):
            arr = [x,y,w,h] = cv2.boundingRect(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            lst.append(arr)
            y0 = lst[i][1] - 3
            if y0 < 0:
                y0 = lst[i][1]
            x0 = lst[i][0]
            yf0 = (y0 + lst[i][3] )
            xf0 = (x0 + lst[i][2] )
            slices.append(image[y0:yf0, x0:xf0 ])
            #   cv2.imshow('card_id', image[y0:yf0, x0:xf0 ]) 
            #   cv2.waitKey(0)      
        return slices
        
    def threshold(self, image):
        gray = cv2.GaussianBlur(image, (3, 3), 2)
        light_less_img = 225 - self.remove_bad_lighting(gray)
        light_less_img *= light_less_img
        segmanted_image = self.segmantation(light_less_img)
        (thresh, thresh_image) = cv2.threshold(segmanted_image, 12, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh_image

        
    def predict_num(self, image, thresh_image,expected_length):
      # plt.imshow(thresh_image)
      # plt.figure()
      clsfInputWidth = self.NIDimage[0]
      clsfInputHeight = self.NIDimage[1]
      space = 0
      nums = []
      _, contours, hierarchy = cv2.findContours(thresh_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      # next line sort contours as perarea size to give me biggest realnumber first 
      # then small noise at last indexes 
      contours.sort(key=lambda x:-cv2.contourArea(x))
      # next line trancat just our desigered numbers  length after sorting them by sizes     
      contours = contours[:expected_length] if expected_length is not '*' else contours
      # next line sorts the numbers as per ther position from left to right       
      contours.sort(key=lambda x:cv2.boundingRect(x))
      print(len(contours))
     
      for indx, contour in enumerate(contours) :     
          boundRect = cv2.boundingRect(contour)
          # if any dimision of countor is biger than desired input for the classifier 
          # we will adjust currant box to contain new dimension of countor then will resize it 
          # this helps us to prevent any bad scaling 
          # if biger than box 
          if (boundRect[2] > clsfInputWidth) | (boundRect[3] > clsfInputHeight)  :
              # select max then adjust classifier box 
              clsfInputWidth = max(boundRect[2] , boundRect[3])
              clsfInputHeight = clsfInputWidth
          else:
                  clsfInputWidth = self.NIDimage[0]
                  clsfInputHeight = self.NIDimage[1]
          
          # find the center of the contour 
          centerPointX = (boundRect[0] + ( boundRect[2] // 2))
          centerPointY = (boundRect[1] + ( boundRect[3] // 2))
          # pointer to the start of the box relative to center of the object 
          x0= centerPointX-(clsfInputWidth//2) - space 
          y0= centerPointY-(clsfInputHeight//2)- space
          # to prevent negative value for each number near to image boundary
          x0 = x0 if x0 > -1 else 0
          y0 = y0 if y0 > -1 else 0
          # the end point to the box that contains the object from start point to box size 
          xf0 = (x0+ clsfInputWidth + space)
          yf0 = (y0 + clsfInputHeight + space)
          # p= (w/2)+s
          # x = p(b/2)
          slice_img = image[y0:yf0 , x0:xf0 ]
          #slice_img = self.heatMapFilter(slice_img) 

          imresize = cv2.resize(slice_img, (self.NIDimage[0] , self.NIDimage[1]))
          # channel last 
          imreshaped = imresize.reshape(self.NIDimage[0] , self.NIDimage[1], 1)
          classes = self.classifier.predict_classes(np.array([imreshaped]))
          nums.append(classes[0])
          cv2.imwrite('./test/'+str(classes[0])+'-'+side+'-'+str(time.time())+'.jpg' , imreshaped)
      return nums
    

    def extract(self, image, expected_length):
        plt.imshow(image)
        plt.figure()
        grayIamge = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # next line to be tested
        grayIamge = imutils.resize(grayIamge, height=100)
        plt.imshow(grayIamge)
        plt.figure()
        thresh_image = self.threshold(grayIamge)
        plt.imshow(thresh_image)
        plt.figure()
        arr = self.predict_num(grayIamge, thresh_image, expected_length)
        print(arr)
        return arr
        
    def remove_bad_lighting(self, input_img):
        median = cv2.medianBlur(input_img, 41)
        return  (input_img / median)
        
    def segmantation(self, input_img):
        K = 2
        channels = 3 if len(input_img.shape)> 2 else  1
        Z = input_img.reshape((-1,channels))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10, cv2.KMEANS_PP_CENTERS )
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        thresh_segmanted_image = res.reshape((input_img.shape))
        return thresh_segmanted_image 
        
    def dodge(self, image, mask):
        """
        
        """
        return cv2.divide(image, 255-mask, scale=256)
        
    
    def error_handling(self, value):
        """
        
        """
        if value is not None:
            return value
        else:
            return False

    def heatMapFilter(self, number_image):
        """
        This Function maps gray scale input image into HeatMap (COLORMAP_JET)
        
        Args:
            number_image (2D Array): gray scale image
        
        Returns:
            whiteBkgNumber (2D Array):Binary Image, white background and black number
        """
        contourAreas = []
        solidBlk = np.zeros(number_image.shape, np.int8)
        heatmapImage = cv2.applyColorMap(number_image, cv2.COLORMAP_JET) 
        filtered = cv2.inRange(heatmapImage,(254,0,0),(255,255,0))
        _, contours, hierarchy = cv2.findContours(filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            contourAreas.append(cv2.contourArea(c))
        cv2.drawContours(solidBlk, contours, contourAreas.index(max(contourAreas)), 255, -1)
        whiteBkgNumber = cv2.inRange(solidBlk,0,100)

        return whiteBkgNumber

    def store_csv(self, data):
        """
        This Function Stores front and back side data in CSV file
        
        Args:
            data (dic): front and back side extracted data
        """
        is_file_exist = os.path.exists(self.file_name)
        with open(self.file_name ,'a') as csvfile:
            field_names = list(data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            if not is_file_exist:
                writer.writeheader()
            writer.writerow(data)
            
    