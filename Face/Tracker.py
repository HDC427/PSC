import cv2
import pickle
import faces_train
from fonctions.fonction import dedans


class Tracker:
    num_pers = 2
    
    def __init__(self, videopath):
        self.num_pers = Tracker.num_pers
        self.cap = cv2.VideoCapture(videopath)
        self.frame = None
        
        self.person0 = Person('Person', '0')
        self.person1 = Person('Person', '1')
        
        self.persons = [self.person0, self.person1]
        
        self.detector = None
        self.recognizer = None

        
    def initialise(self):
        
        face_train.entraine_visage()
        # dont know where are id_s in the detector model must be in label ?
        self.person0.put_face_id(0)
        self.person1.put_face_id(1)
        
    def load_model(self):
        
        self.detector = cv2.CascadeClassifier('C:/Users/Adriano/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')    
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("face-trainner.yml")
        
        for person in persons:
            person.load_models(detector, recognizer)
    
    def update(self):
        
        _,self.frame = self.cap.read()
        
        squel_0, conf_0 = self.person0.identifie(self.image, keypoint_coords)
        squel_1, conf_1 = self.person1.identifie(self.image, keypoint_coords)
    
        # Should we use link_confidences in case of squel conflicts
        # is there face conflict in the method ?
        # conf_0, conf_1 = link_confidences([confidences_0, confidences_1])
        
        self.person0.update(squel_0, conf_0)
        self.person1.update(squel_1, conf_1)
        
        return self.frame
        
    def draw(self):
        
        self.person0.draw(self.frame, 4)
        self.person1.draw(self.frame, 2)
        return self.frame