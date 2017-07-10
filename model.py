from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import numpy as np
import cv2
import glob

class Model(object):

    def __init__(self):
        self.svc = LinearSVC(C=0.0001)
        self.features_scaler = StandardScaler()
        self.training_time = None
        self.testing_accuracy = None

    def fit(self,features,labels):
        self.features_scaler.fit(features)
        s_features = self.features_scaler.transform(features)
        rand_state = 50  #np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(s_features, labels, test_size=0.10, random_state=rand_state)
        t1=time.time()
        self.svc.fit(X_train, y_train)
        t2=time.time()
        self.testing_accuracy = self.svc.score(X_test, y_test)
        self.training_time = round(t2-t1, 2)

    def predict(self,features):
        s_features = self.features_scaler.transform(np.array(features).reshape(1, -1))
        return self.svc.predict(s_features)