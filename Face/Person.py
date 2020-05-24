# -*- coding: utf-8 -*-

"""

Created on Sat Apr  4 19:05:45 2020



@author: Adriano

"""

import Face

from random import randint as rand

class Person:

    def __init__(self,fst,snd):

        self.fst = fst

        self.snd = snd

        self.color = (rand(0,255),rand(0,255),rand(0,255))

        self.Face = Face.Face()

        self.kp = None

        self.cf = 0

    def put_face_id(self, id):
        
        self.Face.set_face_id(id)
        
        
    def load_models(self, detection_model, identification_model):
        
        self.Face.load_model(detection_model, identification_model)
        

    def update(self,keypoint, conf):

        self.kp = keypoint
        self.cf = conf

    #Return the kp and the conf corresponding to this person   

    def identifie_face(self,image,keypoint_coords):

        return self.Face.identifie_face(image,keypoint_coords)
    
    def draw(self):
        # draws the squeleton
        return 
