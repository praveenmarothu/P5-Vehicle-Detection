import os
import pickle
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from global_vars import GV
from model import Model
from feature_extractor import FeatureExtractor as FE


class ModelTrainer(object):

    model=None
    features=None
    labels=None
    v_images=[]
    n_images=[]

    @classmethod
    def load_images(cls):
        t1=time.time()
        for filename in glob.iglob('data/non-vehicles/**/*.png', recursive=True):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cls.n_images.append(img)
        for filename in glob.iglob('data/vehicles/**/*.png', recursive=True):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cls.v_images.append(img)
        t2=time.time()
        print("Loading time - " ,round(t2-t1, 2) )
    @classmethod
    def get_features_labels(cls):
        v_features = []
        n_features = []
        t1=time.time()
        for img in cls.v_images:
            sfeatures = FE.extract_spatial_features(img)
            hfeatures = FE.extract_histogram_features(img)
            gfeatures = FE.extract_hog_features(img)
            features=np.concatenate( (sfeatures,hfeatures,gfeatures[0].ravel(),gfeatures[1].ravel(),gfeatures[2].ravel()) )
            v_features.append(features)
        for img in cls.n_images:
            sfeatures = FE.extract_spatial_features(img)
            hfeatures = FE.extract_histogram_features(img)
            gfeatures = FE.extract_hog_features(img)
            features=np.concatenate( (sfeatures,hfeatures,gfeatures[0].ravel(),gfeatures[1].ravel(),gfeatures[2].ravel()) )
            n_features.append(features)

        cls.features = np.vstack((v_features, n_features)).astype(np.float64)
        cls.labels = np.hstack((np.ones(len(v_features)), np.zeros(len(n_features))))
        t2=time.time()
        print("Feature & Label time - " ,round(t2-t1, 2) )

    @classmethod
    def train_model(cls):
        cls.model=Model()
        cls.model.fit(cls.features,cls.labels)
        print("Training Time - " , cls.model.training_time)
        print("Testing Accuracy - " , cls.model.testing_accuracy)

    @classmethod
    def save_model(cls):
        with open('model.p', 'wb') as f:
            pickle.dump(cls.model,f)


    @classmethod
    def train(cls):
        cls.load_images()
        cls.get_features_labels()
        cls.train_model()
        cls.save_model()


    @classmethod
    def get_trained_model(cls):

        model = None
        if cls.model is not None:
            return cls.model
        elif os.path.isfile("model.p"):
            with open('model.p', 'rb') as f:
                cls.model = pickle.load(f)
        else:
            cls.train()
            cls.save_model()

        return cls.model

