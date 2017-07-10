from collections import deque
import random
from multiprocessing.pool import Pool

import cv2
from scipy.ndimage import label
from global_vars import GV
from feature_extractor import FeatureExtractor
import numpy as np
from matplotlib import pyplot as plt
class CarDetector(object):

    pview_debug=0
    hot_debug=0

    heatmaps_history = deque(maxlen=15)
    perspective_views=None

    hot_windows=None
    current_heatmap=None
    image=None
    frame_image=None
    model=None

    @classmethod
    def process(cls,model,image):
        cls.model=model
        cls.frame_image=image
        cls.image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

        cls.get_perspective_views()
        cls.find_hot_windows()
        cls.create_heatmap()
        cls.draw_cars()

        return cls.frame_image


    @classmethod
    def find_hot_windows(cls):
        cls.hot_windows=[]
        for pview in cls.perspective_views:
            #cv2.rectangle(cls.frame_image, (pview[0],pview[1]) ,(pview[2],pview[3]), (0,255,0), 1)
            cls.sliding_window(pview)

    @classmethod
    def create_heatmap(cls):

        heatmap = np.zeros_like(cls.image[:,:,0])
        for win in cls.hot_windows:
            heatmap[win[0][1]:win[1][1],win[0][0]:win[1][0]] += 1
            #cv2.rectangle(cls.frame_image, win[0] ,win[1], (221, 253, 0), 1)

        #cv2.imwrite("output_images/aaa1.png",cv2.cvtColor(cls.frame_image,cv2.COLOR_RGB2BGR))

        #Creating the Current frame heatmap and adding to global heatmaps
        heatmap[heatmap <= 10] = 0
        heatmap[heatmap > 0] = 1



        cls.heatmaps_history.append(heatmap)

        #Creating heatmap from the last 20 video frames
        cls.current_heatmap=np.zeros_like(heatmap)
        for hmap in cls.heatmaps_history:
            cls.current_heatmap += hmap

        #Should be present in atleast 3 frames
        if GV.current_frame > 10 :
            cls.current_heatmap[cls.current_heatmap <= 3] = 0

    @classmethod
    def draw_cars(cls):
        labels = label(cls.current_heatmap)
        for car_number in range(1, labels[1]+1 ):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(cls.frame_image, bbox[0], bbox[1], (221, 253, 0), 2)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.rectangle(cls.frame_image, (bbox[0][0]-1,bbox[0][1]-18), (bbox[0][0]+48,bbox[0][1]), (221, 253, 0), -1) #(44, 130, 201)
            cv2.putText(cls.frame_image, "CAR" ,(bbox[0][0]+3,bbox[0][1]-3), font, 1,(0,0,0),1,cv2.LINE_AA)


    #Does a sliding window on the given perspective view of the image by first resizing it to a 64 pixel height
    @classmethod
    def sliding_window(cls,pview):

        #Perspective View coordinates
        x1,y1,x2,y2=pview

        #Initial height,width & aspect ratio of perspective view
        iheight = y2-y1
        iwidth = x2-x1
        aspect_ratio = iwidth/iheight

        #Resized height,width
        nheight=64
        nwidth=int(nheight*aspect_ratio)

        #Resized Perspective view
        pmg=cv2.resize(cls.image[y1:y2,x1:x2,:],(nwidth,nheight))

        #Hog features of the entire perspective view
        hog_features = FeatureExtractor.extract_hog_features(pmg)

        #Sliding window on perspective view with a sliding of 8 pixel increments
        #Slide size is 8 pixels is to be able to subsample the hog features as it has 8 pixels per block
        img_block_size=64
        slide_size=8
        hog_block_size=8
        xend=nwidth-img_block_size+1
        for idx,vx in enumerate(range(0,xend,slide_size)):

            #Select the sub-image specified by the sliding window
            smg=pmg[:,vx:vx+img_block_size]

            #Get the features for the current sliding window
            sfeatures = FeatureExtractor.extract_spatial_features(smg)
            hfeatures = FeatureExtractor.extract_histogram_features(smg)

            gfeatures0 = hog_features[0][:,idx:idx+hog_block_size].ravel()
            gfeatures1 = hog_features[1][:,idx:idx+hog_block_size].ravel()
            gfeatures2 = hog_features[2][:,idx:idx+hog_block_size].ravel()
            features=np.concatenate( (sfeatures,hfeatures,gfeatures0,gfeatures1,gfeatures2) )
            pred = cls.model.predict(features)
            #bmg=cv2.cvtColor(smg,cv2.COLOR_HSV2BGR)
            #cv2.imwrite("genimgs/img" + str(random.randint(0,1000)) + ".png" ,bmg)

            if(pred == 1 ):
                #Calculating the window coordinates in main image basing on the resized image coordinates
                rx1=int(vx * (iwidth/nwidth)) + x1
                ry1=y1
                rx2=rx1+iheight
                ry2=ry1+iheight
                cls.hot_windows.append( ((rx1,ry1), (rx2,ry2)) )
                #cv2.rectangle(cls.frame_image,(rx1,ry1),(rx2,ry2), (0,255,0), 1)
                if(vx>xend-32):
                    cls.hot_windows.append( ((rx1,ry1), (rx2,ry2)) )
                #if(rx1<730):
                #    bmg=cv2.cvtColor(smg,cv2.COLOR_HSV2BGR)
                #    cv2.imwrite("genimgs/img" + str(random.randint(0,1000)) + ".png" ,bmg)



    @classmethod
    def get_perspective_views(cls):

        if cls.perspective_views is not None:
            return

        #Outer & Inner Perspective View coordinates
        ox1,oy1,ox2,oy2= (410,390,1280,530)
        ix1,iy1,ix2,iy2= (612,410,1000,450)

        #Windows between Outer & Inner Perspective Views
        left=np.geomspace(ox1,ix1,20,endpoint=True).astype("int")
        top=np.geomspace(oy1,iy1,20,endpoint=True).astype("int")
        right=np.geomspace(ox2,ix2,20,endpoint=True).astype("int")
        bottom=np.geomspace(oy2,iy2,20,endpoint=True).astype("int")

        cls.perspective_views = np.column_stack( (left,top,right,bottom) )