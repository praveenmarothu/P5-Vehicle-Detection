import cv2
import numpy as np
from skimage.feature import hog

class FeatureExtractor(object):

    @staticmethod
    def extract_spatial_features(img):
        size = (20,20)
        features = cv2.resize(img, size).ravel()
        return features

    @staticmethod
    def extract_histogram_features(img):
        channel1_hist = np.histogram(img[:,:,0], bins=128, range=(0,256))
        channel2_hist = np.histogram(img[:,:,1], bins=128, range=(0,256))
        channel3_hist = np.histogram(img[:,:,2], bins=128, range=(0,256))
        features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return features

    @staticmethod
    def extract_hog_features(img):
        f0 = hog(img[:,:,0], orientations=12, pixels_per_cell=(8,8),cells_per_block=(1,1),visualise=False, feature_vector=False,block_norm="L2-Hys")
        f1 = hog(img[:,:,1], orientations=12, pixels_per_cell=(8,8),cells_per_block=(1,1),visualise=False, feature_vector=False,block_norm="L2-Hys")
        f2 = hog(img[:,:,2], orientations=12, pixels_per_cell=(8,8),cells_per_block=(1,1),visualise=False, feature_vector=False,block_norm="L2-Hys")
        return [f0,f1,f2]

