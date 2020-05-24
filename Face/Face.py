# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:31:26 2020

@author: Adriano
"""

import cv2
import pickle
import faces_train
from fonctions.fonction import dedans


class Face:

    def __init__(self,id_):
        self.id = -10
        self.detector = None
        self.recognizer = None
        # self.entraine() # Should be in the initialisation field of Track
#        self.face_cascade = cv2.CascadeClassifier('C:/Users/Adriano/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')    
#        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
#        self.recognizer.read("face-trainner.yml")
#        self.labels = {"person_name": 1}
#        with open("face-labels.pickle", 'rb') as f:
#            self.labels = pickle.load(f)
#            self.labels = {v:k for k,v in self.labels.items()}
# What are labels ?

    def put_face_id(self, id):
        
        self.id = id
        
        
    def load_model(self, detection_model, identification_model):
        
        self.detector = detection_model
        self.recognizer = identification_model
        
    
    #Entraine le "reconnaisseur de visage"
    #def entraine():
    #    faces_train.entraine_visage()
    
    #Renvoie l'indice de la personne (keypoint_coords) qui correspond a self.id_ 
    def identifie_face(self,image,keypoint_coords):
       
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, scaleFactor=1.1, minNeighbors=5)
        liste_nez = []
        for i in range(len(keypoint_scores)):
            liste_nez.append((int(keypoint_coords[i][0][1]),int(keypoint_coords[i][0][0])))
        for (x, y, w, h) in faces:

            roi_gray = gray[y:y+h,x:x+w] #(ycord_start, ycord_end)
            for i,nez in enumerate(liste_nez):
                if dedans(x,y,w,h,nez): 
                    id_trouve, conf = self.recognizer.predict(roi_gray)
                    if self.id == id_trouve:
                        return keypoint_coords[i], conf
                    break
                    
            
            
    
 

